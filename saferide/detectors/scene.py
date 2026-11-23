from __future__ import annotations

import logging
from itertools import combinations
from typing import Callable

import cv2
import numpy as np

from ..alarms import Alarm
from ..types import Observation, Person
from .base import Analyzer, Persistence

log = logging.getLogger(__name__)

# Prompt sets for zero-shot CLIP classification. Neutral prompts deliberately avoid naming the
# location (e.g. "school bus"), otherwise CLIP latches onto the setting instead of the activity.
CALM_PROMPTS = [
    "a photo of people sitting calmly",
    "a photo of people talking calmly",
    "a photo of a person walking",
    "a photo of a person using a phone or reading",
]
ALTERCATION_PROMPTS = [
    "a photo of people arguing angrily",
    "a photo of people fighting",
    "a photo of people pushing each other",
    "a photo of a person shouting at another person",
]
SCENE_PROMPTS = {
    "weapon": "a photo of a person holding a gun or knife",
    "vandalism": "a photo of a person vandalizing or breaking property",
    "fire": "a photo of fire or thick smoke",
}

Scorer = Callable[[list[np.ndarray]], np.ndarray]


class ClipModel:
    """Shared zero-shot CLIP model; `scorer(prompts)` builds a classifier over a prompt set."""

    def __init__(self, device: str | None = None, model: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLIP %s (%s) on %s", model, pretrained, self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    def scorer(self, prompts: list[str]) -> Scorer:
        """Returns a function mapping BGR images to softmax probabilities over `prompts`."""
        from PIL import Image

        torch = self.torch
        with torch.no_grad():
            t = self.model.encode_text(self.tokenizer(prompts).to(self.device))
            text = t / t.norm(dim=-1, keepdim=True)

        def score(images: list[np.ndarray]) -> np.ndarray:
            batch = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)))
                                 for i in images]).to(self.device)
            with torch.no_grad():
                f = self.model.encode_image(batch)
                f = f / f.norm(dim=-1, keepdim=True)
                return (100.0 * f @ text.T).softmax(dim=-1).cpu().numpy()

        return score


def pair_crop(frame: np.ndarray, a: Person, b: Person, pad: float = 0.1) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = np.minimum(a.box[:2], b.box[:2])
    x2, y2 = np.maximum(a.box[2:], b.box[2:])
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return frame[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]


def close_pairs(persons: list[Person], proximity: float, max_pairs: int) -> list[tuple[Person, Person]]:
    scored = []
    for a, b in combinations(persons, 2):
        d = float(np.linalg.norm(a.center - b.center)) / ((a.height + b.height) / 2)
        if d <= proximity:
            scored.append((d, a, b))
    scored.sort(key=lambda x: x[0])
    return [(a, b) for _, a, b in scored[:max_pairs]]


class ClipAltercationAnalyzer(Analyzer):
    """Detects altercations (arguments, shoving, fighting) between nearby people.

    Crops the region around each close pair of people and classifies it with CLIP as calm vs.
    aggressive. Unlike the pose heuristic this also catches confrontations without fast strikes.
    """

    name = "altercation"

    def __init__(self, clip: ClipModel | None = None, scorer: Scorer | None = None,
                 every_n: int = 2, proximity: float = 1.0, max_pairs: int = 3,
                 threshold: float = 0.85, window: int = 6, hits: int = 4):
        self.scorer = scorer or (clip or ClipModel()).scorer(CALM_PROMPTS + ALTERCATION_PROMPTS)
        self.n_calm = len(CALM_PROMPTS)
        self.every_n = every_n
        self.proximity = proximity
        self.max_pairs = max_pairs
        self.threshold = threshold
        self.vote = Persistence(window, hits)
        self.n = 0

    def update(self, obs: Observation) -> list[Alarm]:
        self.n += 1
        if self.n % self.every_n:
            return []
        pairs = close_pairs(obs.persons, self.proximity, self.max_pairs)
        crops = [c for c in (pair_crop(obs.frame, a, b) for a, b in pairs) if c.size]
        best, who = 0.0, None
        if crops:
            probs = self.scorer(crops)[:, self.n_calm:].sum(axis=1)
            i = int(np.argmax(probs))
            best, who = float(probs[i]), pairs[i]
        if self.vote.update(best >= self.threshold) and who is not None:
            a, b = who
            return [Alarm(kind=self.name,
                          message=f"Altercation/fight between person #{a.track_id} and "
                                  f"#{b.track_id} ({sum(self.vote.buf)}/{len(self.vote.buf)} "
                                  f"recent checks aggressive)",
                          confidence=best, severity="CRITICAL")]
        return []


class ClipSceneAnalyzer(Analyzer):
    """Whole-frame zero-shot check for weapons, vandalism and fire."""

    name = "scene"

    def __init__(self, clip: ClipModel | None = None, scorer: Scorer | None = None,
                 every_n: int = 5, threshold: float = 0.6, window: int = 4, hits: int = 3):
        self.labels = list(SCENE_PROMPTS)
        self.scorer = scorer or (clip or ClipModel()).scorer(CALM_PROMPTS + list(SCENE_PROMPTS.values()))
        self.n_calm = len(CALM_PROMPTS)
        self.every_n = every_n
        self.threshold = threshold
        self.votes = {k: Persistence(window, hits) for k in self.labels}
        self.n = 0

    def update(self, obs: Observation) -> list[Alarm]:
        self.n += 1
        if self.n % self.every_n:
            return []
        probs = self.scorer([obs.frame])[0][self.n_calm:]
        alarms = []
        for k, p in zip(self.labels, probs):
            if self.votes[k].update(p >= self.threshold):
                alarms.append(Alarm(kind=f"{self.name}:{k}", message=f"Scene looks like '{k}'",
                                    confidence=float(p), severity="HIGH"))
        return alarms

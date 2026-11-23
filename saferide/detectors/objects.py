from __future__ import annotations

import numpy as np

from ..alarms import Alarm
from ..types import L_WRIST, R_WRIST, Observation
from .base import Analyzer, Persistence

# COCO labels (knife, scissors, baseball bat) plus labels common in custom weapon/fire models.
DEFAULT_DANGEROUS: dict[str, str] = {
    "knife": "HIGH",
    "scissors": "MEDIUM",
    "baseball bat": "MEDIUM",
    "gun": "CRITICAL",
    "pistol": "CRITICAL",
    "handgun": "CRITICAL",
    "rifle": "CRITICAL",
    "firearm": "CRITICAL",
    "weapon": "CRITICAL",
    "fire": "CRITICAL",
    "smoke": "HIGH",
}


def _near_hand(box: np.ndarray, obs: Observation, margin: float = 0.15) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    x1, y1 = box[0] - margin * w, box[1] - margin * h
    x2, y2 = box[2] + margin * w, box[3] + margin * h
    for p in obs.persons:
        for idx in (L_WRIST, R_WRIST):
            x, y, c = p.keypoints[idx]
            if c > 0.3 and x1 <= x <= x2 and y1 <= y <= y2:
                return True
    return False


class DangerousObjectAnalyzer(Analyzer):
    name = "dangerous_object"

    def __init__(self, labels: dict[str, str] | None = None, min_conf: float = 0.4,
                 window: int = 8, hits: int = 3):
        self.labels = {k.lower(): v for k, v in (labels or DEFAULT_DANGEROUS).items()}
        self.min_conf = min_conf
        self._votes = {k: Persistence(window, hits) for k in self.labels}

    def update(self, obs: Observation) -> list[Alarm]:
        best: dict[str, tuple[float, np.ndarray]] = {}
        for d in obs.objects:
            if d.label in self.labels and d.conf >= self.min_conf:
                if d.label not in best or d.conf > best[d.label][0]:
                    best[d.label] = (d.conf, d.box)

        alarms = []
        for label, vote in self._votes.items():
            if vote.update(label in best) and label in best:
                conf, box = best[label]
                held = _near_hand(box, obs)
                severity = "CRITICAL" if held else self.labels[label]
                where = "held by a person" if held else "visible in scene"
                alarms.append(Alarm(kind=f"{self.name}:{label}",
                                    message=f"Dangerous item '{label}' {where}",
                                    confidence=conf, severity=severity))
        return alarms

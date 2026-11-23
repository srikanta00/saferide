from __future__ import annotations

import cv2
import numpy as np

from ..alarms import Alarm
from ..types import Observation
from .base import Analyzer, Persistence


class TamperAnalyzer(Analyzer):
    """Detects camera tampering: lens covered/blacked out, defocus/spray, and camera moved.

    Learns a baseline of the scene during a warm-up period and keeps adapting it slowly
    while the view looks normal.
    """

    name = "tamper"

    def __init__(self, warmup: int = 25, dark_thr: float = 18.0, uniform_std: float = 10.0,
                 blur_ratio: float = 0.25, scene_corr: float = 0.45,
                 window: int = 12, hits: int = 9, adapt: float = 0.02):
        self.warmup = warmup
        self.dark_thr = dark_thr
        self.uniform_std = uniform_std
        self.blur_ratio = blur_ratio
        self.scene_corr = scene_corr
        self.adapt = adapt
        self.n = 0
        self.sharp_base: float | None = None
        self.ref: np.ndarray | None = None
        self.votes = {k: Persistence(window, hits) for k in ("covered", "blurred", "moved")}

    @staticmethod
    def _prep(frame: np.ndarray) -> np.ndarray:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _edges(g: np.ndarray) -> np.ndarray:
        e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 150).astype(np.float32)
        return cv2.GaussianBlur(e, (9, 9), 0)

    def update(self, obs: Observation) -> list[Alarm]:
        g = self._prep(obs.frame)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        edges = self._edges(g)
        self.n += 1

        if self.ref is None or self.n <= self.warmup:
            self.sharp_base = sharp if self.sharp_base is None else 0.8 * self.sharp_base + 0.2 * sharp
            self.ref = edges if self.ref is None else 0.8 * self.ref + 0.2 * edges
            return []

        covered = g.mean() < self.dark_thr or g.std() < self.uniform_std
        blurred = not covered and sharp < self.blur_ratio * max(self.sharp_base, 1e-6)
        corr = float(np.corrcoef(edges.ravel(), self.ref.ravel())[0, 1]) \
            if edges.std() > 0 and self.ref.std() > 0 else 0.0
        moved = not covered and not blurred and corr < self.scene_corr

        flags = {"covered": covered, "blurred": blurred, "moved": moved}
        msgs = {
            "covered": "Camera view blocked or blacked out (lens covered?)",
            "blurred": f"Camera image lost focus (sharpness {sharp:.0f} vs baseline "
                       f"{self.sharp_base:.0f}; sprayed/defocused?)",
            "moved": f"Camera view changed significantly (scene similarity {corr:.2f}; "
                     f"camera moved/redirected?)",
        }
        alarms = []
        for k, flag in flags.items():
            if self.votes[k].update(flag):
                alarms.append(Alarm(kind=f"{self.name}:{k}", message=msgs[k],
                                    confidence=1.0, severity="HIGH"))

        if not any(flags.values()):
            a = self.adapt
            self.sharp_base = (1 - a) * self.sharp_base + a * sharp
            self.ref = (1 - a) * self.ref + a * edges
        return alarms

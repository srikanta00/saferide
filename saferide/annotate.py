from __future__ import annotations

import cv2
import numpy as np

from .types import Observation

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def annotate(frame: np.ndarray, obs: Observation | None, labels: list[str] | None = None) -> np.ndarray:
    """Returns a copy of `frame` with people/object boxes and alarm labels drawn on it."""
    img = frame.copy()
    if obs is not None:
        for p in obs.persons:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(img, f"#{p.track_id}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
        for d in obs.objects:
            if d.label == "person":  # already drawn from pose tracking
                continue
            x1, y1, x2, y2 = map(int, d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
            cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    for i, text in enumerate(labels or []):
        cv2.putText(img, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    return img

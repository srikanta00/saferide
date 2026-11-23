from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 keypoint indices used by the analyzers
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class Person:
    track_id: int
    box: np.ndarray            # [x1, y1, x2, y2]
    keypoints: np.ndarray      # (17, 3) -> x, y, conf
    conf: float = 1.0

    @property
    def height(self) -> float:
        return float(max(self.box[3] - self.box[1], 1.0))

    @property
    def width(self) -> float:
        return float(max(self.box[2] - self.box[0], 1.0))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


@dataclass
class Detection:
    label: str
    conf: float
    box: np.ndarray


@dataclass
class Observation:
    frame: np.ndarray
    timestamp: float
    persons: list[Person] = field(default_factory=list)
    objects: list[Detection] = field(default_factory=list)

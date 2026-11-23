from __future__ import annotations

import logging
import time

import numpy as np

from .types import Detection, Observation, Person

log = logging.getLogger(__name__)


class Perception:
    """Runs YOLO pose tracking (people) and YOLO object detection on each frame."""

    def __init__(self, pose_model: str = "yolov8n-pose.pt", object_model: str = "yolov8s.pt",
                 device: str | None = None, imgsz: int = 640,
                 person_conf: float = 0.4, object_conf: float = 0.35):
        from ultralytics import YOLO  # heavy import kept local

        log.info("Loading pose model %s and object model %s", pose_model, object_model)
        self.pose = YOLO(pose_model)
        self.det = YOLO(object_model)
        self.device = device
        self.imgsz = imgsz
        self.person_conf = person_conf
        self.object_conf = object_conf

    def reset(self) -> None:
        """Forget tracked people (call when switching to a different camera)."""
        for model in (self.pose, self.det):
            for tracker in getattr(getattr(model, "predictor", None), "trackers", None) or []:
                tracker.reset()

    def __call__(self, frame: np.ndarray) -> Observation:
        obs = Observation(frame=frame, timestamp=time.time())
        kw = dict(imgsz=self.imgsz, verbose=False)
        if self.device:
            kw["device"] = self.device

        pose_res = self.pose.track(frame, persist=True, conf=self.person_conf,
                                   tracker="bytetrack.yaml", **kw)[0]
        if pose_res.boxes is not None and len(pose_res.boxes):
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            confs = pose_res.boxes.conf.cpu().numpy()
            ids = (pose_res.boxes.id.cpu().numpy().astype(int)
                   if pose_res.boxes.id is not None else -np.arange(1, len(boxes) + 1))
            kpts = pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints is not None \
                else np.zeros((len(boxes), 17, 3))
            for b, c, i, k in zip(boxes, confs, ids, kpts):
                obs.persons.append(Person(track_id=int(i), box=b, keypoints=k, conf=float(c)))

        det_res = self.det.predict(frame, conf=self.object_conf, **kw)[0]
        names = det_res.names
        if det_res.boxes is not None:
            for b, c, cls in zip(det_res.boxes.xyxy.cpu().numpy(),
                                 det_res.boxes.conf.cpu().numpy(),
                                 det_res.boxes.cls.cpu().numpy().astype(int)):
                obs.objects.append(Detection(label=str(names[cls]).lower(), conf=float(c), box=b))
        return obs

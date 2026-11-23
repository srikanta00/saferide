from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

from .detectors import CrowdAnalyzer, DangerousObjectAnalyzer, FightAnalyzer, TamperAnalyzer
from .detectors.base import Analyzer
from .detectors.objects import DEFAULT_DANGEROUS

log = logging.getLogger("saferide")


@dataclass
class EngineConfig:
    pose_model: str = "yolov8n-pose.pt"
    object_model: str = "yolov8s.pt"
    device: str | None = None
    imgsz: int = 640
    max_fps: float = 8.0
    cooldown: float = 15.0
    max_people: int = 0
    extra_dangerous: list[str] = field(default_factory=list)
    no_tamper: bool = False
    no_clip: bool = False
    clip_scene: bool = False
    altercation_threshold: float = 0.85

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EngineConfig":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in vars(args).items() if k in names})


def add_detection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pose-model", default="yolov8n-pose.pt", help="YOLO pose weights (people + keypoints)")
    p.add_argument("--object-model", default="yolov8s.pt",
                   help="YOLO detection weights; use a custom weapon/fire model for guns, fire, etc.")
    p.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps ...")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-fps", type=float, default=8.0, help="Max frames analysed per second")
    p.add_argument("--cooldown", type=float, default=15.0, help="Seconds before repeating an alarm of the same kind")
    p.add_argument("--max-people", type=int, default=0, help="Crowd alarm above this many people (0=off)")
    p.add_argument("--extra-dangerous", nargs="*", default=[],
                   help="Additional object labels to treat as dangerous (from your custom model)")
    p.add_argument("--no-tamper", action="store_true", help="Disable camera tamper detection")
    p.add_argument("--no-clip", action="store_true",
                   help="Disable CLIP-based altercation detection (arguments/shoving/fighting between nearby people)")
    p.add_argument("--clip-scene", action="store_true",
                   help="Also run whole-frame CLIP check for weapons, vandalism and fire")
    p.add_argument("--altercation-threshold", type=float, default=0.85,
                   help="CLIP aggression probability (0-1) for altercation; raise to reduce false alarms")


class Engine:
    """Loads the AI models once and creates fresh per-camera analyzers on demand."""

    def __init__(self, cfg: EngineConfig):
        from .perception import Perception

        self.cfg = cfg
        self.perception = Perception(cfg.pose_model, cfg.object_model, device=cfg.device, imgsz=cfg.imgsz)
        self.clip = None
        if not cfg.no_clip or cfg.clip_scene:
            from .detectors.scene import ClipModel
            self.clip = ClipModel(device=cfg.device)
        self.dangerous = dict(DEFAULT_DANGEROUS)
        self.dangerous.update({lbl.lower(): "HIGH" for lbl in cfg.extra_dangerous})

    def make_analyzers(self) -> list[Analyzer]:
        cfg = self.cfg
        analyzers: list[Analyzer] = [DangerousObjectAnalyzer(self.dangerous), FightAnalyzer()]
        if not cfg.no_tamper:
            analyzers.append(TamperAnalyzer())
        if cfg.max_people > 0:
            analyzers.append(CrowdAnalyzer(cfg.max_people))
        if self.clip is not None:
            from .detectors.scene import ClipAltercationAnalyzer, ClipSceneAnalyzer
            if not cfg.no_clip:
                analyzers.append(ClipAltercationAnalyzer(self.clip, threshold=cfg.altercation_threshold))
            if cfg.clip_scene:
                analyzers.append(ClipSceneAnalyzer(self.clip))
        return analyzers

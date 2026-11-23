from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime

import cv2

from .alarms import Alarm, AlarmManager
from .detectors import (CrowdAnalyzer, DangerousObjectAnalyzer, FightAnalyzer,
                        TamperAnalyzer)
from .detectors.base import Analyzer
from .stream import RTSPStream, redact
from .types import Observation

log = logging.getLogger("saferide")


class SafeRide:
    def __init__(self, perception, analyzers: list[Analyzer], alarms: AlarmManager,
                 snapshot_dir: str | None = None):
        self.perception = perception
        self.analyzers = analyzers
        self.alarms = alarms
        self.snapshot_dir = snapshot_dir
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)

    def process(self, frame) -> list[Alarm]:
        obs: Observation = self.perception(frame)
        raised = []
        for a in self.analyzers:
            try:
                for alarm in a.update(obs):
                    if self.alarms.raise_alarm(alarm):
                        raised.append(alarm)
            except Exception:  # one faulty analyzer must not stop monitoring
                log.exception("Analyzer %s failed", a.name)
        if raised and self.snapshot_dir:
            self._snapshot(obs, raised)
        return raised

    def _snapshot(self, obs: Observation, raised: list[Alarm]) -> None:
        img = obs.frame.copy()
        for p in obs.persons:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"#{p.track_id}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for d in obs.objects:
            if d.label == "person":  # already drawn from pose tracking
                continue
            x1, y1, x2, y2 = map(int, d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for i, a in enumerate(raised):
            cv2.putText(img, a.kind, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        kinds = "_".join(a.kind.replace(":", "-").replace(" ", "") for a in raised)
        path = os.path.join(self.snapshot_dir, f"{stamp}_{kinds}.jpg")
        cv2.imwrite(path, img)
        log.info("Snapshot saved: %s", path)

    def run(self, stream: RTSPStream, max_fps: float = 8.0) -> None:
        last_id, period = 0, 1.0 / max_fps if max_fps > 0 else 0.0
        frames, t_stat = 0, time.time()
        while True:
            t0 = time.time()
            fid, frame = stream.read()
            if frame is None or fid == last_id:
                if stream.ended:
                    log.info("End of video")
                    return
                time.sleep(0.01)
                continue
            last_id = fid
            self.process(frame)
            frames += 1
            if time.time() - t_stat >= 30:
                log.info("Processing at %.1f FPS", frames / (time.time() - t_stat))
                frames, t_stat = 0, time.time()
            time.sleep(max(0.0, period - (time.time() - t0)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI surveillance alarm system for RTSP IP cameras")
    p.add_argument("source", help="RTSP URL (rtsp://user:pass@host:554/stream), video file, or webcam index")
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
    p.add_argument("--snapshot-dir", default=None, help="Save annotated frames when an alarm fires")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .detectors.objects import DEFAULT_DANGEROUS
    from .perception import Perception

    perception = Perception(args.pose_model, args.object_model, device=args.device, imgsz=args.imgsz)
    dangerous = dict(DEFAULT_DANGEROUS)
    dangerous.update({lbl.lower(): "HIGH" for lbl in args.extra_dangerous})

    analyzers: list[Analyzer] = [DangerousObjectAnalyzer(dangerous), FightAnalyzer()]
    if not args.no_tamper:
        analyzers.append(TamperAnalyzer())
    if args.max_people > 0:
        analyzers.append(CrowdAnalyzer(args.max_people))
    if not args.no_clip or args.clip_scene:
        from .detectors.scene import ClipAltercationAnalyzer, ClipModel, ClipSceneAnalyzer
        clip = ClipModel(device=args.device)
        if not args.no_clip:
            analyzers.append(ClipAltercationAnalyzer(clip, threshold=args.altercation_threshold))
        if args.clip_scene:
            analyzers.append(ClipSceneAnalyzer(clip))

    saferide = SafeRide(perception, analyzers, AlarmManager(cooldown_s=args.cooldown), args.snapshot_dir)
    stream = RTSPStream(args.source).start()
    log.info("Monitoring %s with analyzers: %s", redact(args.source), ", ".join(a.name for a in analyzers))
    try:
        saferide.run(stream, max_fps=args.max_fps)
    except KeyboardInterrupt:
        log.info("Stopping")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime

import cv2

from .alarms import Alarm, AlarmManager
from .annotate import annotate
from .detectors.base import Analyzer
from .engine import Engine, EngineConfig, add_detection_args
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
        self.last_obs: Observation | None = None
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)

    def process(self, frame) -> list[Alarm]:
        obs: Observation = self.perception(frame)
        self.last_obs = obs
        accepted = []
        for a in self.analyzers:
            try:
                accepted += [alarm for alarm in a.update(obs) if self.alarms.accept(alarm)]
            except Exception:  # one faulty analyzer must not stop monitoring
                log.exception("Analyzer %s failed", a.name)
        if accepted and self.snapshot_dir:
            name = self._snapshot(obs, accepted)
            for alarm in accepted:
                alarm.snapshot = name
        for alarm in accepted:  # dispatched after the snapshot exists so listeners can link to it
            self.alarms.emit(alarm)
        return accepted

    def _snapshot(self, obs: Observation, raised: list[Alarm]) -> str:
        img = annotate(obs.frame, obs, [a.kind for a in raised])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        kinds = "_".join(a.kind.replace(":", "-").replace(" ", "") for a in raised)
        name = f"{stamp}_{kinds}.jpg"
        cv2.imwrite(os.path.join(self.snapshot_dir, name), img)
        log.info("Snapshot saved: %s", os.path.join(self.snapshot_dir, name))
        return name

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
    add_detection_args(p)
    p.add_argument("--snapshot-dir", default=None, help="Save annotated frames when an alarm fires")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = EngineConfig.from_args(args)
    engine = Engine(cfg)
    analyzers = engine.make_analyzers()
    saferide = SafeRide(engine.perception, analyzers, AlarmManager(cooldown_s=cfg.cooldown), args.snapshot_dir)
    stream = RTSPStream(args.source).start()
    log.info("Monitoring %s with analyzers: %s", redact(args.source), ", ".join(a.name for a in analyzers))
    try:
        saferide.run(stream, max_fps=cfg.max_fps)
    except KeyboardInterrupt:
        log.info("Stopping")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()

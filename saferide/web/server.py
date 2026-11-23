from __future__ import annotations

import argparse
import logging

from ..engine import Engine, EngineConfig, add_detection_args

DEFAULT_PORT = 8765


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m saferide.web",
                                description="SafeRide web interface (set the camera URL, watch the stream, see alarms)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Interface to listen on. Keep 127.0.0.1 unless you add authentication in front of it.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--snapshot-dir", default="alarms", help="Where alarm snapshots are saved and served from")
    p.add_argument("--camera", default=None, help="Optional RTSP URL to start monitoring immediately")
    p.add_argument("-v", "--verbose", action="store_true")
    add_detection_args(p)
    return p


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from ..monitor import Monitor
    from .app import create_app

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("saferide")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Listening on %s: anyone who can reach this port can control the camera and see "
                    "the video. There is no login; put it behind an authenticating reverse proxy or VPN.",
                    args.host)

    engine = Engine(EngineConfig.from_args(args))
    monitor = Monitor(engine, snapshot_dir=args.snapshot_dir)
    app = create_app(monitor, args.snapshot_dir)
    if args.camera:
        monitor.start(args.camera)
    log.info("SafeRide web interface on http://%s:%d", args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()

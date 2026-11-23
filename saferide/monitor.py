from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from urllib.parse import urlsplit

import cv2

from .alarms import Alarm, AlarmManager
from .annotate import annotate
from .main import SafeRide
from .stream import RTSPStream, redact

log = logging.getLogger("saferide.monitor")

ALLOWED_SCHEMES = ("rtsp", "rtsps")
BANNER_SECONDS = 6.0
STALE_AFTER_S = 10.0


class InvalidCameraURL(ValueError):
    pass


def validate_camera_url(url: str) -> str:
    """Only RTSP(S) URLs are accepted, so the web UI cannot be used to make the server open
    arbitrary local files, devices or other protocols."""
    url = (url or "").strip()
    if not url or len(url) > 2048 or any(c.isspace() for c in url):
        raise InvalidCameraURL("Enter a camera URL such as rtsp://user:pass@192.168.1.10:554/stream1")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise InvalidCameraURL("Only rtsp:// and rtsps:// URLs are supported")
    try:
        host = parts.hostname
        parts.port  # raises ValueError on a bad port
    except ValueError:
        raise InvalidCameraURL("The URL has an invalid port") from None
    if not host:
        raise InvalidCameraURL("The URL has no host name")
    return url


class AlarmStore:
    """Thread-safe, bounded, id-ordered alarm history."""

    def __init__(self, maxlen: int = 500):
        self._items: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._next_id = 1

    def add(self, alarm: Alarm) -> dict:
        with self._lock:
            item = {
                "id": self._next_id,
                "kind": alarm.kind,
                "message": alarm.message,
                "severity": alarm.severity,
                "confidence": round(alarm.confidence, 3),
                "timestamp": alarm.timestamp,
                "time": datetime.fromtimestamp(alarm.timestamp).isoformat(timespec="seconds"),
                "snapshot": f"/snapshots/{alarm.snapshot}" if alarm.snapshot else None,
            }
            self._next_id += 1
            self._items.append(item)
            return item

    def latest_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    def list(self, limit: int = 100) -> list[dict]:
        """Newest first."""
        with self._lock:
            return list(self._items)[::-1][:limit]

    def since(self, last_id: int) -> list[dict]:
        """Alarms newer than `last_id`, oldest first."""
        with self._lock:
            return [i for i in self._items if i["id"] > last_id]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


@dataclass
class _Session:
    url: str
    stream: object
    stop: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    last_frame_at: float | None = None
    process_times: deque = field(default_factory=lambda: deque(maxlen=12))
    error: str | None = None
    banners: dict[str, float] = field(default_factory=dict)
    safe: SafeRide | None = None


class Monitor:
    """Runs one camera at a time in background threads.

    - the processing thread runs the AI pipeline (a few frames per second);
    - the publishing thread streams every frame to the browser with the latest detections drawn.
    """

    def __init__(self, engine, snapshot_dir: str | None = None,
                 stream_factory: Callable[[str], object] = RTSPStream,
                 publish_fps: float = 15.0, jpeg_width: int = 960, jpeg_quality: int = 75):
        self.engine = engine
        self.snapshot_dir = snapshot_dir
        self.stream_factory = stream_factory
        self.publish_period = 1.0 / publish_fps
        self.jpeg_width = jpeg_width
        self.jpeg_quality = jpeg_quality
        self.alarms = AlarmStore()
        self._lock = threading.Lock()  # serialises start/stop
        self._session: _Session | None = None
        self._session_no = 0
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._jpeg_cond = threading.Condition()

    # ---- control -------------------------------------------------------------------------
    def start(self, url: str) -> None:
        url = validate_camera_url(url)
        with self._lock:
            self._stop_locked()
            if hasattr(self.engine.perception, "reset"):
                self.engine.perception.reset()
            self._session_no += 1
            session = _Session(url=url, stream=self.stream_factory(url))
            self._session = session
            session.stream.start()
            session.threads = [
                threading.Thread(target=self._process_loop, args=(session,), daemon=True, name="saferide-process"),
                threading.Thread(target=self._publish_loop, args=(session,), daemon=True, name="saferide-publish"),
            ]
            for t in session.threads:
                t.start()
            log.info("Monitoring %s", redact(url))

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        session.stop.set()
        for t in session.threads:
            t.join(timeout=30)
        session.stream.stop()
        with self._jpeg_cond:
            self._jpeg = None
            self._jpeg_seq += 1
            self._jpeg_cond.notify_all()
        log.info("Stopped monitoring %s", redact(session.url))

    # ---- state ---------------------------------------------------------------------------
    def status(self) -> dict:
        s = self._session
        if s is None:
            return {"state": "idle", "url": None, "fps": 0.0, "error": None, "session": self._session_no}
        now = time.time()
        if s.last_frame_at is None:
            state = "connecting"
        elif now - s.last_frame_at > STALE_AFTER_S:
            state = "reconnecting"
        else:
            state = "running"
        times = list(s.process_times)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        if state != "running":
            fps = 0.0
        return {"state": state, "url": redact(s.url), "fps": round(fps, 1), "error": s.error,
                "session": self._session_no}

    def jpeg(self) -> tuple[int, bytes | None]:
        with self._jpeg_cond:
            return self._jpeg_seq, self._jpeg

    # ---- worker threads ------------------------------------------------------------------
    def _on_alarm(self, session: _Session, alarm: Alarm) -> None:
        print(alarm, flush=True)
        self.alarms.add(alarm)
        session.banners[alarm.kind] = time.time() + BANNER_SECONDS

    def _process_loop(self, session: _Session) -> None:
        safe = SafeRide(self.engine.perception, self.engine.make_analyzers(),
                        AlarmManager(cooldown_s=self.engine.cfg.cooldown,
                                     sink=lambda a: self._on_alarm(session, a)),
                        self.snapshot_dir)
        session.safe = safe
        period = 1.0 / self.engine.cfg.max_fps if self.engine.cfg.max_fps > 0 else 0.0
        last_id = 0
        while not session.stop.is_set():
            t0 = time.time()
            fid, frame = session.stream.read()
            if frame is None or fid == last_id:
                time.sleep(0.01)
                continue
            last_id = fid
            try:
                safe.process(frame)
                session.process_times.append(time.time())
                session.error = None
            except Exception as e:  # keep monitoring; surface the problem in the UI
                log.exception("Processing failed")
                session.error = f"{type(e).__name__}: {e}"
                time.sleep(0.5)
            session.stop.wait(max(0.0, period - (time.time() - t0)))

    def _publish_loop(self, session: _Session) -> None:
        last_id = 0
        while not session.stop.is_set():
            fid, frame = session.stream.read()
            if frame is None or fid == last_id:
                session.stop.wait(0.02)
                continue
            last_id = fid
            t0 = time.time()
            session.last_frame_at = t0
            safe = session.safe
            labels = [k for k, until in list(session.banners.items()) if until > t0]
            img = annotate(frame, safe.last_obs if safe else None, labels)
            h, w = img.shape[:2]
            if w > self.jpeg_width:
                img = cv2.resize(img, (self.jpeg_width, int(h * self.jpeg_width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                with self._jpeg_cond:
                    self._jpeg = buf.tobytes()
                    self._jpeg_seq += 1
                    self._jpeg_cond.notify_all()
            session.stop.wait(max(0.0, self.publish_period - (time.time() - t0)))

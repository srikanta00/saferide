from __future__ import annotations

import logging
import os
import re
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


def redact(url: str) -> str:
    """Hide credentials in rtsp://user:pass@host URLs before logging."""
    return re.sub(r"(://[^:/@]+):[^@]+@", r"\1:***@", str(url))


class RTSPStream:
    """Background reader that always exposes the latest frame and auto-reconnects.

    Reading in a dedicated thread prevents the decoder buffer from filling up
    (which would otherwise make inference run on stale frames).
    """

    def __init__(self, url: str, reconnect_delay_s: float = 2.0, use_tcp: bool = True):
        self.url = url
        self.reconnect_delay_s = reconnect_delay_s
        if use_tcp and str(url).startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.ended = False

    def _open(self) -> cv2.VideoCapture:
        if str(self.url).isdigit():
            cap = cv2.VideoCapture(int(self.url))
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        is_file = os.path.isfile(str(self.url))
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                log.warning("Cannot open stream %s; retrying in %.1fs", redact(self.url), self.reconnect_delay_s)
                time.sleep(self.reconnect_delay_s)
                continue
            log.info("Connected to %s", redact(self.url))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                if is_file:  # emulate real-time playback for local test videos
                    time.sleep(1.0 / fps)
            cap.release()
            if is_file:
                self.ended = True
                return
            log.warning("Stream lost; reconnecting in %.1fs", self.reconnect_delay_s)
            time.sleep(self.reconnect_delay_s)

    def start(self) -> "RTSPStream":
        self._thread.start()
        return self

    def read(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

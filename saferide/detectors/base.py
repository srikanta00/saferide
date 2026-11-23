from __future__ import annotations

from collections import deque

from ..alarms import Alarm
from ..types import Observation


class Analyzer:
    name = "analyzer"

    def update(self, obs: Observation) -> list[Alarm]:
        raise NotImplementedError


class Persistence:
    """Sliding-window vote: fires when at least `hits` of the last `window` updates were positive.

    Filters out single-frame false positives, which are common with CCTV footage.
    """

    def __init__(self, window: int, hits: int):
        self.buf: deque[bool] = deque(maxlen=window)
        self.hits = hits

    def update(self, positive: bool) -> bool:
        self.buf.append(bool(positive))
        return sum(self.buf) >= self.hits

    def reset(self) -> None:
        self.buf.clear()

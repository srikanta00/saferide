from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


@dataclass
class Alarm:
    kind: str
    message: str
    confidence: float
    severity: str = "HIGH"
    timestamp: float = field(default_factory=time.time)
    snapshot: str | None = None  # file name of the annotated frame, when saved

    def __str__(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return (f"[ALARM][{ts}][{self.severity}] {self.kind.upper()}: "
                f"{self.message} (confidence={self.confidence:.2f})")


class AlarmManager:
    """Dispatches alarms, suppressing repeats of the same kind within a cooldown."""

    def __init__(self, cooldown_s: float = 10.0,
                 sink: Callable[[Alarm], None] = lambda a: print(a, flush=True),
                 clock: Callable[[], float] = time.time):
        self.cooldown_s = cooldown_s
        self.sink = sink
        self.clock = clock
        self._last: dict[str, float] = {}

    def accept(self, alarm: Alarm) -> bool:
        """Cooldown check; records the alarm time when accepted but does not dispatch it."""
        now = self.clock()
        last = self._last.get(alarm.kind)
        if last is not None and now - last < self.cooldown_s:
            return False
        self._last[alarm.kind] = now
        alarm.timestamp = now
        return True

    def emit(self, alarm: Alarm) -> None:
        self.sink(alarm)

    def raise_alarm(self, alarm: Alarm) -> bool:
        if not self.accept(alarm):
            return False
        self.emit(alarm)
        return True

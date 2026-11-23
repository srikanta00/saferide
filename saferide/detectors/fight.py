from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np

from ..alarms import Alarm
from ..types import (L_ANKLE, L_HIP, L_SHOULDER, L_WRIST, R_ANKLE, R_HIP, R_SHOULDER,
                     R_WRIST, Observation, Person)
from .base import Analyzer, Persistence

LIMBS = (L_WRIST, R_WRIST, L_ANKLE, R_ANKLE)


class _Track:
    def __init__(self, maxlen: int = 10):
        self.hist: deque[tuple[float, np.ndarray, float]] = deque(maxlen=maxlen)
        self.down = 0
        self.last_seen = 0.0

    def add(self, t: float, p: Person) -> None:
        self.hist.append((t, p.keypoints.copy(), p.height))
        self.last_seen = t

    def limb_speed(self, kp_conf: float = 0.3) -> float:
        """Peak wrist/ankle speed in body-heights per second over the last two samples."""
        if len(self.hist) < 2:
            return 0.0
        (t0, k0, h0), (t1, k1, h1) = self.hist[-2], self.hist[-1]
        dt = max(t1 - t0, 1e-3)
        h = max((h0 + h1) / 2, 1.0)
        best = 0.0
        for i in LIMBS:
            if k0[i, 2] > kp_conf and k1[i, 2] > kp_conf:
                best = max(best, float(np.linalg.norm(k1[i, :2] - k0[i, :2])) / h / dt)
        return best


def _arm_raised_toward(a: Person, b: Person, kp_conf: float = 0.3) -> bool:
    """True if a wrist of `a` is at/above shoulder height and extended towards `b`."""
    for w, s in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        wx, wy, wc = a.keypoints[w]
        sx, sy, sc = a.keypoints[s]
        if wc < kp_conf or sc < kp_conf:
            continue
        reach = abs(wx - sx) / a.height
        toward = np.sign(wx - sx) == np.sign(b.center[0] - a.center[0])
        if wy <= sy + 0.05 * a.height and reach > 0.25 and toward:
            return True
    return False


def _is_lying(p: Person, kp_conf: float = 0.3) -> bool:
    if p.width < 1.2 * p.height:
        return False
    sh = [p.keypoints[i] for i in (L_SHOULDER, R_SHOULDER) if p.keypoints[i, 2] > kp_conf]
    hp = [p.keypoints[i] for i in (L_HIP, R_HIP) if p.keypoints[i, 2] > kp_conf]
    if not sh or not hp:
        return True  # box aspect alone
    torso = np.mean(hp, axis=0)[:2] - np.mean(sh, axis=0)[:2]
    return abs(torso[0]) > abs(torso[1])  # torso more horizontal than vertical


class FightAnalyzer(Analyzer):
    """Pose-based altercation detection.

    A frame is "aggressive" when two people are in close proximity and at least one of
    them shows fast limb motion (strikes/kicks) or an arm raised towards the other.
    A fight alarm requires aggression to persist over a sliding window of frames.
    Also raises a 'person_down' alarm when someone stays lying on the ground.
    """

    name = "fight"

    def __init__(self, proximity: float = 1.0, speed_thr: float = 1.8,
                 window: int = 15, hits: int = 6, down_frames: int = 10,
                 track_ttl_s: float = 3.0):
        self.proximity = proximity
        self.speed_thr = speed_thr
        self.vote = Persistence(window, hits)
        self.down_frames = down_frames
        self.track_ttl_s = track_ttl_s
        self.tracks: dict[int, _Track] = {}

    def _pair_score(self, a: Person, b: Person) -> float:
        dist = float(np.linalg.norm(a.center - b.center)) / ((a.height + b.height) / 2)
        if dist > self.proximity:
            return 0.0
        speed = max(self.tracks[a.track_id].limb_speed(), self.tracks[b.track_id].limb_speed())
        score = 0.0
        if speed > self.speed_thr:
            score += min(speed / (2 * self.speed_thr), 1.0)
        if _arm_raised_toward(a, b) or _arm_raised_toward(b, a):
            score += 0.4
        return min(score, 1.0) * (1.0 - 0.5 * dist / self.proximity)

    def update(self, obs: Observation) -> list[Alarm]:
        t = obs.timestamp
        for p in obs.persons:
            self.tracks.setdefault(p.track_id, _Track()).add(t, p)
        self.tracks = {k: v for k, v in self.tracks.items() if t - v.last_seen <= self.track_ttl_s}

        alarms: list[Alarm] = []
        scores = [self._pair_score(a, b) for a, b in combinations(obs.persons, 2)]
        top = max(scores, default=0.0)
        if self.vote.update(top >= 0.5):
            ratio = sum(self.vote.buf) / len(self.vote.buf)
            alarms.append(Alarm(kind="fight",
                                message=f"Possible fight/altercation between "
                                        f"{len(obs.persons)} people (aggression in "
                                        f"{sum(self.vote.buf)}/{len(self.vote.buf)} recent frames)",
                                confidence=max(top, ratio), severity="CRITICAL"))

        for p in obs.persons:
            tr = self.tracks[p.track_id]
            tr.down = tr.down + 1 if _is_lying(p) else 0
            if tr.down == self.down_frames:
                alarms.append(Alarm(kind="person_down",
                                    message=f"Person #{p.track_id} lying on the ground "
                                            f"(possible injury/fall)",
                                    confidence=p.conf, severity="HIGH"))
        return alarms


class CrowdAnalyzer(Analyzer):
    name = "crowd"

    def __init__(self, max_people: int, window: int = 10, hits: int = 8):
        self.max_people = max_people
        self.vote = Persistence(window, hits)

    def update(self, obs: Observation) -> list[Alarm]:
        n = len(obs.persons)
        if self.vote.update(n > self.max_people):
            return [Alarm(kind="crowd", message=f"Unusual crowding: {n} people "
                                                f"(limit {self.max_people})",
                          confidence=1.0, severity="MEDIUM")]
        return []

import numpy as np
import pytest

from saferide.alarms import Alarm, AlarmManager
from saferide.detectors import DangerousObjectAnalyzer, FightAnalyzer, TamperAnalyzer
from saferide.stream import redact
from saferide.types import (L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST,
                         Detection, Observation, Person)

FRAME = np.zeros((480, 640, 3), np.uint8)


def person(tid, x, y=100, h=200, w=80, wrist_dx=0.0, wrist_y=None):
    k = np.zeros((17, 3))
    cx = x + w / 2
    k[[L_SHOULDER, R_SHOULDER]] = [[cx - 15, y + 40, 1], [cx + 15, y + 40, 1]]
    k[[L_HIP, R_HIP]] = [[cx - 12, y + 110, 1], [cx + 12, y + 110, 1]]
    wy = y + 100 if wrist_y is None else wrist_y
    k[[L_WRIST, R_WRIST]] = [[cx - 20 + wrist_dx, wy, 1], [cx + 20 + wrist_dx, wy, 1]]
    return Person(tid, np.array([x, y, x + w, y + h], float), k)


def test_alarm_cooldown():
    out, now = [], [0.0]
    m = AlarmManager(cooldown_s=10, sink=out.append, clock=lambda: now[0])
    assert m.raise_alarm(Alarm("fight", "x", 1))
    now[0] = 5
    assert not m.raise_alarm(Alarm("fight", "x", 1))
    assert m.raise_alarm(Alarm("tamper", "x", 1))
    now[0] = 11
    assert m.raise_alarm(Alarm("fight", "x", 1))
    assert len(out) == 3


def test_dangerous_object_requires_persistence_and_detects_held():
    a = DangerousObjectAnalyzer(window=5, hits=3)
    knife_box = np.array([110, 180, 150, 220], float)  # around right wrist of person below
    p = person(1, 60)
    obs = lambda: Observation(FRAME, 0, [p], [Detection("knife", 0.8, knife_box)])
    assert a.update(obs()) == [] and a.update(obs()) == []
    alarms = a.update(obs())
    assert [x.kind for x in alarms] == ["dangerous_object:knife"]
    assert alarms[0].severity == "CRITICAL"


def test_dangerous_object_ignores_benign_and_low_conf():
    a = DangerousObjectAnalyzer(window=3, hits=1)
    obs = Observation(FRAME, 0, [], [Detection("cup", 0.9, np.zeros(4)),
                                      Detection("knife", 0.1, np.zeros(4))])
    assert a.update(obs) == []


def test_fight_detected_for_close_people_with_fast_limbs():
    f = FightAnalyzer(window=10, hits=4)
    alarms = []
    for i in range(10):
        dx = 60 if i % 2 else -60  # rapid flailing arms
        t = i * 0.1
        alarms += f.update(Observation(FRAME, t, [person(1, 100, wrist_dx=dx),
                                                  person(2, 190, wrist_dx=-dx)]))
    assert any(a.kind == "fight" for a in alarms)


def test_no_fight_for_calm_or_distant_people():
    calm, far = FightAnalyzer(window=10, hits=4), FightAnalyzer(window=10, hits=4)
    alarms = []
    for i in range(20):
        t, dx = i * 0.1, (60 if i % 2 else -60)
        alarms += calm.update(Observation(FRAME, t, [person(1, 100), person(2, 190)]))
        alarms += far.update(Observation(FRAME, t, [person(1, 0, wrist_dx=dx),
                                                    person(2, 550, wrist_dx=-dx)]))
    assert not any(a.kind == "fight" for a in alarms)


def test_person_down():
    f = FightAnalyzer(down_frames=3)
    lying = Person(7, np.array([100, 300, 400, 380], float), np.zeros((17, 3)))
    kinds = [a.kind for i in range(4) for a in f.update(Observation(FRAME, i * 0.1, [lying]))]
    assert kinds == ["person_down"]


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 120, np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, 600), rng.integers(0, 440)
        img[y:y + 40, x:x + 40] = rng.integers(0, 255, 3)
    return img


def _run_tamper(frames):
    t = TamperAnalyzer(warmup=5, window=4, hits=3)
    out = []
    for f in frames:
        out += [a.kind for a in t.update(Observation(f, 0))]
    return out


def test_tamper_normal_scene_no_alarm():
    assert _run_tamper([_scene()] * 20) == []


@pytest.mark.parametrize("bad,kind", [
    (np.zeros((480, 640, 3), np.uint8), "tamper:covered"),
    (None, "tamper:blurred"),
    (_scene(seed=99), "tamper:moved"),
])
def test_tamper_detects(bad, kind):
    import cv2
    if bad is None:
        bad = cv2.GaussianBlur(_scene(), (51, 51), 0)
    assert kind in _run_tamper([_scene()] * 8 + [bad] * 5)


def test_redact():
    assert redact("rtsp://admin:s3cr3t@10.0.0.5:554/s1") == "rtsp://admin:***@10.0.0.5:554/s1"
    assert redact("rtsp://10.0.0.5/s1") == "rtsp://10.0.0.5/s1"


def _fake_scorer(aggr):
    """Returns 4 calm + 4 aggressive probabilities with total aggression `aggr()` per crop."""
    def score(crops):
        a = aggr()
        return np.array([[(1 - a) / 4] * 4 + [a / 4] * 4 for _ in crops])
    return score


def _altercation_alarms(aggr_values, persons):
    from saferide.detectors.scene import ClipAltercationAnalyzer
    it = iter(aggr_values)
    an = ClipAltercationAnalyzer(scorer=_fake_scorer(lambda: next(it)), every_n=1, window=6, hits=4)
    return [a for _ in aggr_values for a in an.update(Observation(FRAME, 0, persons))]


def test_altercation_requires_sustained_aggression():
    pair = [person(1, 100), person(2, 190)]
    assert _altercation_alarms([0.95, 0.2, 0.9, 0.1, 0.3, 0.2], pair) == []
    alarms = _altercation_alarms([0.9, 0.95, 0.5, 0.92, 0.97], pair)
    assert [a.kind for a in alarms] == ["altercation"]
    assert "#1" in alarms[0].message and "#2" in alarms[0].message


def test_altercation_ignores_people_far_apart():
    assert _altercation_alarms([1.0] * 6, [person(1, 0), person(2, 550)]) == []

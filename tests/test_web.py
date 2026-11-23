import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from saferide.alarms import Alarm
from saferide.detectors.base import Analyzer
from saferide.monitor import AlarmStore, InvalidCameraURL, Monitor, validate_camera_url
from saferide.types import Observation
from saferide.web.app import create_app


class FakeStream:
    def __init__(self, url):
        self.url, self.n, self.stopped = url, 0, False

    def start(self):
        return self

    def read(self):
        time.sleep(0.005)
        self.n += 1
        return self.n, np.full((120, 160, 3), 90, np.uint8)

    def stop(self):
        self.stopped = True


class AlwaysAlarm(Analyzer):
    name = "fake"

    def update(self, obs):
        return [Alarm("fake", "something happened", 0.9, "HIGH")]


class FakeEngine:
    cfg = SimpleNamespace(cooldown=60.0, max_fps=50.0)

    def __init__(self):
        self.perception = lambda frame: Observation(frame=frame, timestamp=time.time())
        self.perception.reset = lambda: None

    def make_analyzers(self):
        return [AlwaysAlarm()]


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def monitor(tmp_path):
    m = Monitor(FakeEngine(), snapshot_dir=str(tmp_path), stream_factory=FakeStream)
    yield m
    m.stop()


@pytest.fixture
def client(monitor, tmp_path):
    return TestClient(create_app(monitor, str(tmp_path), frontend_dist=tmp_path / "missing"))


@pytest.mark.parametrize("url", ["", "   ", "http://host/x", "file:///etc/passwd", "rtsp://", "rtsp://h:99999/x",
                                 "rtsp://host/a b", "0", "/dev/video0"])
def test_validate_rejects_bad_urls(url):
    with pytest.raises(InvalidCameraURL):
        validate_camera_url(url)


def test_validate_accepts_rtsp():
    assert validate_camera_url(" rtsp://u:p@10.0.0.5:554/s1 ") == "rtsp://u:p@10.0.0.5:554/s1"
    assert validate_camera_url("RTSPS://cam.local/x")


def test_alarm_store_orders_limits_and_since():
    s = AlarmStore(maxlen=3)
    for i in range(5):
        s.add(Alarm("k", f"m{i}", 0.5, snapshot="a.jpg" if i == 4 else None))
    assert [a["id"] for a in s.list()] == [5, 4, 3]
    assert [a["id"] for a in s.since(3)] == [4, 5]
    assert s.list()[0]["snapshot"] == "/snapshots/a.jpg" and s.list()[1]["snapshot"] is None
    s.clear()
    assert s.list() == [] and s.latest_id() == 5


def test_api_rejects_non_rtsp(client):
    r = client.post("/api/camera", json={"url": "file:///etc/passwd"})
    assert r.status_code == 400
    assert client.get("/api/status").json()["state"] == "idle"


def test_camera_lifecycle_alarms_and_credential_redaction(client, monitor):
    r = client.post("/api/camera", json={"url": "rtsp://admin:s3cret@10.0.0.5:554/s1"})
    assert r.status_code == 200
    assert "s3cret" not in r.text and "admin:***@" in r.json()["url"]

    assert wait_for(lambda: client.get("/api/alarms").json())
    alarm = client.get("/api/alarms").json()[0]
    assert alarm["kind"] == "fake" and alarm["severity"] == "HIGH"
    assert wait_for(lambda: monitor.status()["state"] == "running")
    assert wait_for(lambda: monitor.jpeg()[1] is not None)
    assert monitor.jpeg()[1][:2] == b"\xff\xd8"  # JPEG magic

    assert client.delete("/api/camera").json()["state"] == "idle"
    assert monitor.jpeg()[1] is None
    assert client.delete("/api/alarms").json() == {"ok": True}
    assert client.get("/api/alarms").json() == []


def test_switching_camera_stops_previous_stream(client, monitor):
    streams = []
    monitor.stream_factory = lambda url: streams.append(FakeStream(url)) or streams[-1]
    client.post("/api/camera", json={"url": "rtsp://a/1"})
    client.post("/api/camera", json={"url": "rtsp://b/2"})
    assert streams[0].stopped and not streams[1].stopped
    assert client.get("/api/status").json()["url"] == "rtsp://b/2"


def test_root_explains_missing_frontend(client):
    assert "not been built" in client.get("/").json()["message"]

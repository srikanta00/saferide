# SafeRide – AI alarm system for IP cameras (RTSP)

Reads a live RTSP feed and prints an alarm when it detects unusual activity.

| Alarm kind | How it is detected |
|---|---|
| `altercation` | **Main fight/argument detector.** For each pair of people close together, the area around them is cropped and scored by CLIP (zero-shot) as calm or aggressive (arguing, shouting, pushing, fighting). It must stay aggressive for 4 of the last 6 checks. This catches confrontations even when nobody throws a punch. |
| `fight` | YOLOv8-pose + ByteTrack. It looks for two people close together where at least one has fast wrist/ankle motion (punches or kicks) or an arm raised toward the other. The behavior must persist across a sliding window of frames. |
| `person_down` | A tracked person who stays lying on the ground (possible injury or fall). |
| `dangerous_object:<label>` | YOLO object detection (COCO: `knife`, `scissors`, `baseball bat`). Severity becomes **CRITICAL** when the item is in a person's hand. |
| `tamper:covered` / `tamper:blurred` / `tamper:moved` | The lens is blacked out or covered, the image is defocused or sprayed, or the camera was turned. Compared against a learned scene baseline. |
| `crowd` | More than `--max-people` people for a sustained period (optional). |
| `scene:<weapon/vandalism/fire>` | Optional whole-frame CLIP check (`--clip-scene`). |

Every alarm needs several confirming frames, which suppresses one-off false positives. Repeats of the same alarm kind are rate-limited by `--cooldown`.

## Install & run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # model weights download automatically on first run
python -m saferide "rtsp://user:pass@192.168.1.10:554/stream1" --snapshot-dir alarms/
```

Example output:
```
[ALARM][2026-11-22 15:42:55][MEDIUM] CROWD: Unusual crowding: 4 people (limit 3) (confidence=1.00)
[ALARM][2026-11-22 15:43:09][HIGH] TAMPER:COVERED: Camera view blocked or blacked out (lens covered?) (confidence=1.00)
```

Useful options (`python -m saferide -h`):
- `--device cuda` runs on a GPU. On CPU, expect about 2–3 FPS with CLIP enabled (tune `--max-fps` and `--imgsz`).
- `--object-model weapons.pt --extra-dangerous gun pistol` adds a custom YOLO model. **COCO has no gun or fire class**, so a weapon/fire-trained model is needed for those. Labels such as `gun`, `pistol`, `rifle`, `weapon`, `fire`, and `smoke` are already recognised as dangerous.
- `--altercation-threshold 0.9` reduces altercation false alarms (default `0.85`). `--no-clip` disables CLIP entirely, which is faster but leaves only the punch/kick pose heuristic for fights.
- `--clip-scene` adds the whole-frame weapon/vandalism/fire check.
- `--max-people N` enables the crowd alarm. `--no-tamper` disables tamper checks.
- The source can also be a video file or a webcam index, which is useful for testing.

Credentials in the RTSP URL are masked in the logs.

## Web interface

A browser UI where you set the camera URL, watch the live annotated video and see alarms (with snapshots) appear in real time. The SafeRide web server uses **port 8765** by default (change it with `--port`).

```
 Browser ──► React page ──► FastAPI (saferide.web) ──► RTSP camera
              (frontend/)    /api/*  /stream.mjpg  /snapshots/*
```

| Endpoint | Purpose |
|---|---|
| `POST /api/camera` `{"url": "rtsp://..."}` | Start monitoring a camera (replaces the current one) |
| `DELETE /api/camera` | Stop monitoring |
| `GET /api/status` | `idle`, `connecting`, `running` or `reconnecting`, plus analysed fps (the URL is returned with the password masked) |
| `GET /stream.mjpg` | Live video with detections drawn on it |
| `GET /api/alarms`, `DELETE /api/alarms` | Alarm history (newest first, last 500) / clear it |
| `GET /api/events` | Server-Sent Events stream of new alarms |
| `GET /snapshots/<file>` | Alarm snapshot images |
| `GET /api/docs` | Interactive API docs |

### Production (one server, no Node.js needed to run it)

Build the web page once, then the Python server serves everything on a single port.

```bash
# one-time setup
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# build the web page (needs Node.js 20.19+ or 22.12+ only on the machine that builds it)
cd frontend && npm install && npm run build && cd ..

# run
python -m saferide.web                  # open http://localhost:8765
python -m saferide.web --port 9000      # use another port
python -m saferide.web --camera "rtsp://user:pass@192.168.1.10:554/stream1"   # start a camera immediately
```

`frontend/dist/` is not committed. If you copy the project to a machine without Node.js, copy the built `frontend/dist/` folder along with it.

**Installing Node.js:** the version in Ubuntu's `apt` (18.x) is too old for the build tool. Use [nvm](https://github.com/nvm-sh/nvm) instead, which needs no `sudo`:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# reopen the terminal, then:
nvm install 22
```
All detection options from the command-line tool also work here (`--device cuda`, `--max-people 5`, `--no-clip`, `--object-model ...`). Alarm snapshots are saved in `alarms/` (change with `--snapshot-dir`) and are never deleted automatically.

### Development (two servers, hot reload)

Run these in two terminals. The Vite dev server (port **5173**) serves the React page with hot reload and forwards `/api`, `/stream.mjpg` and `/snapshots` to the Python server (port **8765**).

```bash
# terminal 1: Python API server
. .venv/bin/activate
python -m saferide.web

# terminal 2: React dev server (needs Node.js 20.19+ or 22.12+, see "Installing Node.js" above)
cd frontend
npm install        # first time only
npm run dev        # open http://localhost:5173
```

If the Python server uses a different port, tell Vite where it is: `SAFERIDE_API=http://127.0.0.1:9000 npm run dev`. To change the dev page port, edit `server.port` in `frontend/vite.config.ts`.

### Web security notes
- The server listens on `127.0.0.1` only by default. **There is no login.** Anyone who can reach the port can start cameras and view the video, so if you use `--host 0.0.0.0` put it behind a VPN or an authenticating reverse proxy (a warning is printed).
- Only `rtsp://` and `rtsps://` URLs are accepted, so the page cannot be used to open local files or other protocols. A camera URL still makes the server connect to whatever host you type, which is another reason to restrict access.
- Camera passwords are never sent back to the browser or written to the logs, and the page does not store the URL.
- One camera is monitored at a time; starting another one replaces it. Alarm history is kept in memory and is lost when the server restarts (snapshot images stay on disk).

## Customising alarms
`AlarmManager(sink=...)` accepts any callable. Swap `print` for e-mail, MQTT, webhook, or GPIO buzzer output. New detectors subclass `saferide.detectors.base.Analyzer` and implement `update(obs) -> list[Alarm]`.

## Tests
```bash
python -m pytest -q tests                     # backend (no models or camera needed)
cd frontend && npm run build                  # type-checks and builds the web page
```

## Limitations
The `fight` detector is a pose/motion heuristic aimed at strikes. The CLIP `altercation` detector is broader but zero-shot, so verify it on your own cameras. Dancing, hugging, or sports can trigger it, so tune `FightAnalyzer(speed_thr, proximity, window, hits)` for your camera. For production accuracy, train a dedicated action-recognition model on violence datasets such as RWF-2000 and plug it in as another `Analyzer`.

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

## Customising alarms
`AlarmManager(sink=...)` accepts any callable. Swap `print` for e-mail, MQTT, webhook, or GPIO buzzer output. New detectors subclass `saferide.detectors.base.Analyzer` and implement `update(obs) -> list[Alarm]`.

## Tests
```bash
python -m pytest -q tests
```

## Limitations
The `fight` detector is a pose/motion heuristic aimed at strikes. The CLIP `altercation` detector is broader but zero-shot, so verify it on your own cameras. Dancing, hugging, or sports can trigger it, so tune `FightAnalyzer(speed_thr, proximity, window, hits)` for your camera. For production accuracy, train a dedicated action-recognition model on violence datasets such as RWF-2000 and plug it in as another `Analyzer`.

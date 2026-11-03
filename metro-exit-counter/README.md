# Metro Exit Counter v2

AI people-counting for metro / station exits. YOLOv8 detects people, a ByteTrack-style tracker gives each
person a stable ID, and everyone whose feet enter the **exit zone** is counted exactly once.
v2 adds a live web dashboard, a draggable exit-zone editor, CSV/JSON reports, browser-playable H.264 output,
and a fully portable / Docker-ready packaging.

## Quick start

You need **Python 3.10+** and internet **once**. No `pip install` by hand.

| OS | Start |
|---|---|
| Windows | double-click `start.bat` (or `python run.py`) |
| Linux / macOS | `./start.sh` (or `python3 run.py`) |

First launch downloads the libraries (~100 MB) into the `vendor/` folder automatically. After that it starts instantly and offline.
The dashboard opens at <http://127.0.0.1:8000>. Pick a video (or drag one onto the player), drag the yellow
zone to fit your exit, press **Start analysis**.

Headless: `python run.py --cli my_video.mp4` -> results in `data/outputs/`.

## Copy to another machine - nothing to install

After the first launch the whole folder is self-contained (libraries in `vendor/`, model in `models/`).
Copy/paste the folder to another computer and run `start.bat` / `start.sh` - no pip, no internet.
This works when the other machine has the **same OS, CPU type (x86-64/ARM) and Python minor version (e.g. 3.12)**,
because Python libraries contain compiled code. If it doesn't match, `run.py` detects it and rebuilds `vendor/` for that machine
(needs internet once). `python setup_portable.py` does the same by hand - it is just `pip install -r requirements.txt --target vendor`.

Want one folder that works on several Python versions / both Windows and Linux offline?
`python tools/build_vendor.py win_amd64:311,312,313 linux_x86_64:312,313` pre-builds all of them into `vendor/` (about 300 MB).

## Docker

Docker removes even the Python requirement - the image carries Python, the libraries and the model.

    docker compose up --build        # first build downloads ~150 MB, later starts are instant
    # open http://localhost:8000
    docker compose down              # stop

Plain docker, without compose:

    docker build -t metro-exit-counter .
    docker run -d -p 8000:8000 -v "$PWD/data:/app/data" --name mec metro-exit-counter

* `./data` is mounted into the container, so uploads and results survive restarts.
* Ship the image to another machine: `docker save metro-exit-counter | gzip > mec.tar.gz` ... then `docker load < mec.tar.gz` there.
* Batch mode in Docker: `docker run --rm -v "$PWD/data:/app/data" metro-exit-counter python run.py --cli /app/data/uploads/clip.mp4`
* Webcams/RTSP: pass a URL or camera index as the CLI source (webcams need `--device /dev/video0` on Linux).

How the Dockerfile works: `python:3.12-slim` base -> copy `requirements.txt` and `pip install` (cached layer) -> copy code + model ->
run as non-root -> `HEALTHCHECK` hits `/api/health` -> `CMD python run.py`. `.dockerignore` keeps `vendor/` and your videos out of the image.

## How it works

1. `detector.py` - YOLOv8n exported to ONNX, run with ONNX Runtime (no PyTorch -> ~100 MB instead of ~3 GB).
2. `tracker.py` - two-stage ByteTrack-style association (confident boxes first, weak boxes rescue lost tracks).
   A track must live `min_hits` frames before it gets an ID, which removes flicker/ghost counts.
3. `counter.py` - the foot point (bottom-centre of the box) entering the polygon counts that ID once.
4. `pipeline.py` - runs the above per frame in a background thread, draws overlays, encodes H.264, writes reports.
5. `server.py` + `static/` - Flask API and the offline dashboard (no CDN, no internet needed).

Outputs per run in `data/outputs/`: `run_*.mp4` (annotated video), `run_*.csv` (one row per exit with timestamp), `run_*.json` (summary + timeline).

## Settings (Settings button in the dashboard)

| Setting | Meaning |
|---|---|
| Detection confidence | Higher = fewer false people, may miss some |
| Processing width | 640 = fastest, 1280 = better for small/far people |
| Min. frames to confirm | Frames a person must be seen before being counted |
| Play in real time | Throttle to the video's own speed (demo mode) |

Environment variables: `MEC_PORT`, `MEC_HOST`, `MEC_DATA_DIR`, `MEC_MODEL`.

## Using another model
`python tools/export_onnx.py path/to/model.pt` (needs `pip install ultralytics onnx` once) writes the `.onnx` next to it. Must be a COCO-style YOLOv8 detector where class 0 = person.

## Project layout

    run.py  start.bat  start.sh  setup_portable.py
    Dockerfile  docker-compose.yml  requirements.txt
    metro_counter/   detector, tracker, counter, render, pipeline, server, static/ (dashboard)
    models/          yolov8n.onnx (used) + yolov8n.pt (original)
    samples/         demo_walkthrough.mp4 (synthetic test clip, only to try the UI)
    tests/           python tests/test_core.py
    vendor/          libraries, created on first launch
    legacy_main_original.py   the original single-file script, kept for reference

## Troubleshooting
* **"A required library could not be loaded"** -> vendor does not match this machine; run `python setup_portable.py`.
* **Windows: onnxruntime DLL error** -> install the "Microsoft Visual C++ Redistributable 2015-2022 (x64)".
* **Result video won't play in the page** -> use the Video download button; the file is standard H.264.
* **Slow** -> lower Processing width to 640. CPU-only by design.
* Accuracy depends on camera angle: place the zone where people's *feet* are visible.

Licence: see `LICENSE` (original project by tubakhxn).

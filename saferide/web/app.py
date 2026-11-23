from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..monitor import InvalidCameraURL, Monitor

DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
BOUNDARY = "frame"


class CameraRequest(BaseModel):
    url: str


def create_app(monitor: Monitor, snapshot_dir: str, frontend_dist: Path | None = None) -> FastAPI:
    app = FastAPI(title="SafeRide", docs_url="/api/docs", openapi_url="/api/openapi.json")
    os.makedirs(snapshot_dir, exist_ok=True)

    @app.get("/api/status")
    def status():
        return monitor.status()

    @app.post("/api/camera")
    def set_camera(req: CameraRequest):
        try:
            monitor.start(req.url)
        except InvalidCameraURL as e:
            raise HTTPException(status_code=400, detail=str(e))
        return monitor.status()

    @app.delete("/api/camera")
    def stop_camera():
        monitor.stop()
        return monitor.status()

    @app.get("/api/alarms")
    def list_alarms(limit: int = Query(100, ge=1, le=500)):
        return monitor.alarms.list(limit)

    @app.delete("/api/alarms")
    def clear_alarms():
        monitor.alarms.clear()
        return {"ok": True}

    @app.get("/api/events")
    async def events(request: Request):
        """Server-Sent Events: pushes each new alarm as it is raised."""
        last = request.headers.get("last-event-id")
        last_id = int(last) if last and last.isdigit() else monitor.alarms.latest_id()

        async def gen():
            nonlocal last_id
            idle = 0.0
            yield "retry: 2000\n\n"
            while not await request.is_disconnected():
                new = monitor.alarms.since(last_id)
                for item in new:
                    last_id = item["id"]
                    yield f"id: {item['id']}\nevent: alarm\ndata: {json.dumps(item)}\n\n"
                if new:
                    idle = 0.0
                else:
                    idle += 0.25
                    if idle >= 15:  # keep proxies from closing an idle connection
                        yield ": keepalive\n\n"
                        idle = 0.0
                await asyncio.sleep(0.25)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        """Live annotated video as multipart JPEG (plays in a plain <img> tag)."""
        async def gen():
            last_seq = -1
            while not await request.is_disconnected():
                seq, jpeg = monitor.jpeg()
                if seq != last_seq and jpeg is not None:
                    last_seq = seq
                    yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                           f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"
                else:
                    last_seq = seq
                    await asyncio.sleep(0.03)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    app.mount("/snapshots", StaticFiles(directory=snapshot_dir), name="snapshots")

    dist = frontend_dist if frontend_dist is not None else DEFAULT_FRONTEND_DIST
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return JSONResponse({
                "message": "SafeRide API is running, but the web page has not been built.",
                "fix": "cd frontend && npm install && npm run build  (or use the Vite dev server)",
                "api_docs": "/api/docs",
            })

    return app

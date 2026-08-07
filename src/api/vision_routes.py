"""HTTP surface for the monitoring UI: cameras, live frames, and alerts.

Kept in its own router rather than bolted onto `routes.py` because it has nothing to do
with the template's chat agent — it is the camera product, and mixing them would make both
harder to read.

Two streaming choices worth stating. Video goes out as **MJPEG** (`multipart/x-mixed-replace`),
which an `<img src>` renders natively — no player, no WebSocket, no codec negotiation, and
it degrades to "the last frame" instead of erroring when a camera stalls. Alerts go out as
**SSE**, because they are one-directional server→client and reconnect automatically; a
WebSocket would need its own keepalive and retry logic for no gain.
"""

import asyncio
import json
import os
import uuid

import cv2
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from src.models.camera_schemas import Alert, CameraCreate, CameraInfo, DemoVideo, SystemStatus
from src.services.monitoring import UPLOAD_DIR, MonitoringService

router = APIRouter(tags=["monitoring"])

_ALLOWED_UPLOAD_EXT = {".mp4", ".avi", ".mov", ".mkv"}
# Uploads are read into memory by the frame-probe below, and this is a demo box.
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# 1x1 grey JPEG, served when a camera has not produced a frame yet. An empty body makes
# the browser render a broken-image icon, which reads as a crash rather than "starting".
_PLACEHOLDER_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffc2000b080001000101011100ffc40014000100000000000000000"
    "00000000000000009ffda0008010100000000013fffd9"
)


def get_service(request: Request) -> MonitoringService:
    """Pull the singleton created in `main.lifespan` off app state.

    One service per process, not per request: it owns live camera threads.
    """
    service = getattr(request.app.state, "monitoring", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is not running")
    return service


# ---------------------------------------------------------------- sources
@router.get("/demo-videos", response_model=list[DemoVideo])
async def list_demo_videos() -> list[DemoVideo]:
    """Sample clips bundled with the repo, for the 'Prepared Demo' source option."""
    return MonitoringService.list_demo_videos()


@router.post("/uploads")
async def upload_video(file: UploadFile) -> dict[str, str]:
    """Store an operator-supplied MP4 and return the name to use as a camera source."""
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in _ALLOWED_UPLOAD_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported video type '{extension}'")

    # The stored name is generated, never taken from the client: an uploaded filename is
    # attacker-controlled and would otherwise be a path-traversal write.
    stored_name = f"{uuid.uuid4().hex}{extension}"
    destination = os.path.join(UPLOAD_DIR, stored_name)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    size = 0
    with open(destination, "wb") as handle:
        # Chunked so a large file never has to fit in memory at once.
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                handle.close()
                os.remove(destination)
                raise HTTPException(status_code=413, detail="Video exceeds the 200 MB limit")
            handle.write(chunk)

    return {"filename": stored_name, "original_name": file.filename or stored_name}


@router.get("/preview-frame")
async def preview_frame(
    source_type: str = Query(..., pattern="^(demo|upload|rtsp)$"),
    source: str = Query(...),
) -> Response:
    """Grab one frame from a source so the setup screen can draw a polygon over it.

    The zone canvas needs a still of the actual camera view — drawing on a blank rectangle
    would produce a polygon that means nothing in the scene.
    """
    resolved = MonitoringService.resolve_source(source_type, source)
    capture = cv2.VideoCapture(int(resolved) if resolved.isdigit() else resolved)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise HTTPException(status_code=404, detail=f"Could not read a frame from '{source}'")

    encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not encoded:
        raise HTTPException(status_code=500, detail="Failed to encode the preview frame")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------- cameras
@router.get("/cameras", response_model=list[CameraInfo])
async def list_cameras(request: Request) -> list[CameraInfo]:
    return get_service(request).list_cameras()


@router.post("/cameras", response_model=CameraInfo, status_code=201)
async def create_camera(payload: CameraCreate, request: Request) -> CameraInfo:
    """Register a camera and start analysing it immediately."""
    service = get_service(request)
    try:
        # Opening a capture and spawning threads is blocking work; off the event loop it
        # goes, or every other request stalls behind a slow RTSP handshake.
        return await asyncio.to_thread(service.add_camera, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/cameras/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, request: Request) -> Response:
    service = get_service(request)
    if not await asyncio.to_thread(service.remove_camera, camera_id):
        raise HTTPException(status_code=404, detail=f"No camera '{camera_id}'")
    return Response(status_code=204)


@router.get("/cameras/{camera_id}/snapshot")
async def camera_snapshot(camera_id: str, request: Request) -> Response:
    """Latest analysed frame as a single JPEG — the polling fallback for MJPEG."""
    frame = get_service(request).latest_jpeg(camera_id)
    return Response(
        content=frame or _PLACEHOLDER_JPEG,
        media_type="image/jpeg",
        # Without this a browser happily serves the first frame forever from cache.
        headers={"Cache-Control": "no-store"},
    )


@router.get("/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str, request: Request, fps: float = Query(default=10.0, gt=0, le=30)) -> Response:
    """MJPEG stream of the anonymized, annotated frames for one camera."""
    service = get_service(request)
    if service.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail=f"No camera '{camera_id}'")

    boundary = "frame"
    interval = 1.0 / fps

    async def generate():
        last_sent: bytes | None = None
        while True:
            # Client-gone detection: without it, a closed browser tab leaves this loop
            # spinning until the server is restarted.
            if await request.is_disconnected():
                break

            frame = service.latest_jpeg(camera_id) or _PLACEHOLDER_JPEG
            # Re-sending an identical buffer wastes bandwidth when the pipeline is slower
            # than the requested stream rate, which it usually is.
            if frame is not last_sent:
                last_sent = frame
                yield (
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------- alerts
@router.get("/alerts", response_model=list[Alert])
async def list_alerts(request: Request, unacknowledged_only: bool = False) -> list[Alert]:
    return get_service(request).list_alerts(unacknowledged_only=unacknowledged_only)


@router.post("/alerts/{alert_id}/acknowledge", status_code=204)
async def acknowledge_alert(alert_id: str, request: Request) -> Response:
    if not get_service(request).acknowledge(alert_id):
        raise HTTPException(status_code=404, detail=f"No alert '{alert_id}'")
    return Response(status_code=204)


@router.get("/events")
async def event_stream(request: Request) -> StreamingResponse:
    """Server-sent events: alert created / caption ready / acknowledged.

    This is what makes the modal interruptive without the frontend polling for it.
    """
    service = get_service(request)
    queue = service.subscribe()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # A comment line keeps proxies and browsers from treating an idle
                    # stream as dead and tearing it down.
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {message['type']}\ndata: {json.dumps(message['data'])}\n\n"
        finally:
            service.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        # X-Accel-Buffering off: nginx would otherwise buffer the stream and delay alerts.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/monitoring/status", response_model=SystemStatus)
async def monitoring_status(request: Request) -> SystemStatus:
    return get_service(request).status()

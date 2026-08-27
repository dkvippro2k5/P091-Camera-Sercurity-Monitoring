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
import time
import uuid

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse

from src.api.deps import (
    current_websocket_manager,
    require_guard,
    require_manager,
    require_operator,
    session_is_live,
)
from src.models.camera_schemas import (
    Alert,
    AlertVerdict,
    CameraCreate,
    CameraInfo,
    DemoVideo,
    SimilarIncident,
    SystemStatus,
    NotificationSettings,
    NotificationToggle,
)
from src.services.eval_report import build_report
from src.services.monitoring import UPLOAD_DIR, MonitoringService
from src.services.user_store import User

# Every route here requires a signed-in operator, manager or guard. The dependency is on the
# router rather than per route, so an endpoint added later is protected because of where it
# was put, not because its author remembered to decorate it.
router = APIRouter(tags=["monitoring"], dependencies=[Depends(require_guard)])

# The statistics report is the one manager-only surface. Separate router, and the two never
# share a path, so FastAPI cannot resolve a request against the wrong gate.
manager_router = APIRouter(tags=["monitoring"], dependencies=[Depends(require_manager)])
operator_router = APIRouter(tags=["monitoring"], dependencies=[Depends(require_operator)])
manager_ws_router = APIRouter(tags=["monitoring"])

# How often a long-lived stream re-reads its session from the store, rather than on every
# wake. Bounds how long a revoked session can keep a stream open, traded against a database
# round trip per event — see the comment in `event_stream`'s generator.
_SESSION_RECHECK_SECONDS = 15.0

_ALLOWED_UPLOAD_EXT = {".mp4", ".avi", ".mov", ".mkv"}
# Keep the application limit explicit even though nginx also rejects oversized requests:
# direct backend traffic must enforce the same boundary.
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024

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
# All three exist only to build a camera, so they sit on the same gate as `POST /cameras`
# itself — which is the guard's, since the guard is the one who configures cameras.
#
# Worth naming rather than leaving implicit: these are the interesting half of this app's
# attack surface. `/uploads` writes an operator-supplied file to disk and `/preview-frame`
# opens an arbitrary RTSP URL from the server. That risk did not disappear when the routes
# moved; it is accepted, because the role that builds cameras cannot build them without it.
# Splitting them off from `POST /cameras` would only produce the worse arrangement: the
# probing and uploading reachable by someone who cannot create the camera it is all for.
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
                raise HTTPException(status_code=413, detail="Video exceeds the 500 MB limit")
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
    """Remove a camera and its zone configuration."""
    service = get_service(request)
    if not await asyncio.to_thread(service.remove_camera, camera_id):
        raise HTTPException(status_code=404, detail=f"No camera '{camera_id}'")
    return Response(status_code=204)


@router.post("/cameras/{camera_id}/replay", response_model=CameraInfo)
async def replay_camera(camera_id: str, request: Request) -> CameraInfo:
    """Replay an uploaded or prepared video after it reaches EOF."""
    service = get_service(request)
    try:
        camera = await asyncio.to_thread(service.replay_camera, camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if camera is None:
        raise HTTPException(status_code=404, detail=f"No camera '{camera_id}'")
    return camera


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
    """MJPEG stream of the annotated frames for one camera."""
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
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
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


@router.get("/evidence/{evidence_id}")
async def incident_evidence(evidence_id: str, request: Request) -> FileResponse:
    """Serve local evidence only through the authenticated operator surface."""
    if get_service(request).evidence_store.backend != "local":
        raise HTTPException(status_code=404, detail="Local evidence storage is disabled")
    path = get_service(request).evidence_store.local_path(evidence_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.post("/alerts/{alert_id}/acknowledge", status_code=204)
async def acknowledge_alert(alert_id: str, request: Request) -> Response:
    """Dismiss an alert without judging it.

    Deliberately writes no history row: an unlabelled row would sit in the dataset the
    false-alarm rate is computed from, counted as neither a hit nor a miss. Use
    `/verdict` for the HITL path.
    """
    if not get_service(request).acknowledge(alert_id):
        raise HTTPException(status_code=404, detail=f"No alert '{alert_id}'")
    return Response(status_code=204)


@router.post("/alerts/{alert_id}/verdict", status_code=204)
async def submit_verdict(
    alert_id: str,
    payload: AlertVerdict,
    request: Request,
    guard: User = Depends(require_guard),
) -> Response:
    """The guard's HITL judgement (PRD §3.2), which also becomes training context.

    This is the only path that appends to the labelled history, so it is both the
    operator's answer and the system's feedback loop. The judgement is attributed to the
    signed-in operator: a labelled dataset whose labels have no author cannot be audited
    when two guards disagree about what a false alarm looks like.
    """
    # Taken from the router's own `require_guard` rather than calling `current_user` again:
    # FastAPI caches a dependency's result per request, so declaring the *same* callable as a
    # parameter reuses the gate that already ran instead of re-authenticating. Resolving a
    # session means two Postgres round trips, and this route was paying for them twice.
    judged_by = guard.username
    # Writes to the incident/history store, which is a real network round trip on Postgres
    # (see auth_routes.login's identical concern for scrypt) -- run off the event loop so a
    # slow or stalled database connection cannot freeze every other camera/SSE connection.
    recorded = await asyncio.to_thread(
        get_service(request).record_operator_verdict,
        alert_id,
        payload.verdict,
        judged_by=judged_by,
        operator_note=payload.operator_note,
        actual_event_type=payload.actual_event_type,
    )
    if not recorded:
        raise HTTPException(status_code=404, detail=f"No alert '{alert_id}'")
    return Response(status_code=204)


@router.get("/alerts/{alert_id}/similar", response_model=list[SimilarIncident])
async def similar_incidents(
    alert_id: str, request: Request, limit: int = Query(default=5, ge=1, le=20)
) -> list[SimilarIncident]:
    """Past guard-judged incidents whose rule-engine reason reads similarly to this one's.

    PLANNED feature (docs/database/schema.sql): empty until EMBEDDINGS_ENABLED and
    DATABASE_URL is Postgres, and until at least one alert has been judged since -- the
    comparison library only contains incidents a guard has already confirmed, so a fresh
    deployment starts with nothing to compare against. Never a 404/500 for "not configured
    yet" or "no matches" -- both read as an empty list, the same way a search with zero
    results is not an error.
    """
    rows = await asyncio.to_thread(get_service(request).similar_incidents, alert_id, limit)
    return [SimilarIncident(**row) for row in rows]


@router.get("/events")
async def event_stream(request: Request, guard: User = Depends(require_guard)) -> StreamingResponse:
    """Server-sent events: alert created / triage verdict ready / acknowledged / auth revoked.

    This is what makes the modal interruptive without the frontend polling for it.
    `alert.created` fires within one frame time of the detection; `alert.verdict` follows
    seconds later when the VLM has ruled, so the operator is never made to wait on an API
    call to learn there has been a fall.

    Auth is checked by the router's dependency *before* this runs, so a dead session is a
    plain 401 and no stream is ever opened for it — which matters more here than elsewhere,
    because `EventSource` reconnects on its own and an unauthenticated stream that opened
    and then failed would become an endless reconnect loop.
    """
    service = get_service(request)
    # Reuses the router's `require_guard` result rather than re-resolving the session — see
    # `submit_verdict` for why that is two round trips nobody needs to pay twice.
    username = guard.username
    queue = service.subscribe()

    async def generate():
        # The loop below wakes once per *event*, not once per interval, so during an alert
        # storm across several cameras it can spin many times a second. Re-reading the
        # session from Postgres on each of those wakes was costing a ~0.3 s round trip and a
        # worker thread every time, which is what saturated the threadpool and made the whole
        # dashboard stop responding. The revocation check is a safety net on a timescale of
        # seconds, not milliseconds, so pacing it is free: a logout still tears this stream
        # down within `_SESSION_RECHECK_SECONDS`, and `auth.revoked` below usually gets there
        # first anyway.
        next_session_check = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    break

                # Re-checked periodically, not just at connect. This response can stay open
                # for hours, so without it logging out would leave the alert stream running
                # until the browser felt like closing it — "logout ends the session" has to
                # be something the server enforces, not something the client cooperates with.
                # Off the event loop: on PostgresSessionStore this is a real network call.
                now = time.monotonic()
                if now >= next_session_check:
                    if not await asyncio.to_thread(session_is_live, request):
                        break
                    next_session_check = now + _SESSION_RECHECK_SECONDS

                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # A comment line keeps proxies and browsers from treating an idle
                    # stream as dead and tearing it down.
                    yield ": keepalive\n\n"
                    continue

                # The fan-out has no per-user index, so a revocation reaches every stream and
                # is filtered here. Without the check, one guard logging out would sign out
                # everyone else watching.
                if message["type"] == "auth.revoked":
                    if message["data"].get("username") != username:
                        continue
                    yield f"event: auth.revoked\ndata: {json.dumps(message['data'])}\n\n"
                    break

                yield f"event: {message['type']}\ndata: {json.dumps(message['data'])}\n\n"
        finally:
            service.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        # X-Accel-Buffering off: nginx would otherwise buffer the stream and delay alerts.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@operator_router.get("/notifications", response_model=NotificationSettings)
async def notification_settings(request: Request) -> NotificationSettings:
    return NotificationSettings(**get_service(request).notification_status())


@operator_router.patch("/notifications", response_model=NotificationSettings)
async def toggle_notifications(payload: NotificationToggle, request: Request) -> NotificationSettings:
    return NotificationSettings(**get_service(request).set_notifications_enabled(payload.enabled))


@operator_router.get("/monitoring/status", response_model=SystemStatus)
async def monitoring_status(request: Request) -> SystemStatus:
    return get_service(request).status()


# ---------------------------------------------------------------- manager reports
@manager_router.get("/statistics")
async def statistics(request: Request) -> dict:
    """Both halves of the manager's numbers, in one response.

    - `report` — the PRD §4 evaluation report (`EvalReport`): precision, the rules-alone vs
      rules+VLM comparison, and the caveats. Built by the same code the CLI uses, so the
      dashboard and `scripts/report_eval.py` can never disagree.
    - `operational` — live activity: how many alerts each camera raised, how many are still
      unreviewed, and how long judgements take. The evaluation report deliberately knows
      nothing about this — it is computed from *reviewed* rows only (see its module
      docstring), so totals and timeliness have to come from the incident log instead.
    """
    service = get_service(request)

    # build_report does blocking file I/O; off the event loop, like every other such call.
    report = await asyncio.to_thread(build_report)
    operational = await asyncio.to_thread(service.operational_stats)
    heatmap = await asyncio.to_thread(service.heatmap)

    body = report.model_dump()
    # Nulls are kept deliberately — no `exclude_none`. `precision` is `None` when nothing has
    # been reviewed yet, which `_precision` goes out of its way to distinguish from `0.0`
    # ("every alert was wrong"). Dropping the key would erase that distinction at the very
    # last step and leave the dashboard unable to tell the two apart.
    #
    # `recall_note` is a `@property`, which Pydantic excludes from `model_dump()`. It carries
    # the report's central decision — *why* there is no recall number — so it is re-attached
    # explicitly or the most important statement in the report silently vanishes from the API.
    body["recall_note"] = report.recall_note
    return {"report": body, "operational": operational, "heatmap": heatmap}


@manager_ws_router.websocket("/manager/live")
async def manager_live(websocket: WebSocket) -> None:
    """Push manager dashboard changes without making the browser poll continuously."""
    user = await asyncio.to_thread(current_websocket_manager, websocket)
    if user is None:
        await websocket.close(code=1008, reason="Manager authentication required")
        return

    service = getattr(websocket.app.state, "monitoring", None)
    if service is None:
        await websocket.close(code=1013, reason="Monitoring service is not ready")
        return

    await websocket.accept()
    queue = service.subscribe()
    try:
        await websocket.send_json({"type": "manager.connected", "data": {"username": user.username}})
        # Paced for the same reason as the SSE stream: this loop wakes on every published
        # event, and a session read is a database round trip, not a dict lookup.
        next_session_check = 0.0
        while True:
            now = time.monotonic()
            if now >= next_session_check:
                if not await asyncio.to_thread(session_is_live, websocket):
                    await websocket.close(code=1008, reason="Session expired")
                    return
                next_session_check = now + _SESSION_RECHECK_SECONDS
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json(message)
            except TimeoutError:
                # A bounded heartbeat keeps camera health moving on the manager screen even
                # when no incident event has been published.
                await websocket.send_json({"type": "manager.status", "data": service.status().model_dump()})
    except WebSocketDisconnect:
        return
    finally:
        service.unsubscribe(queue)

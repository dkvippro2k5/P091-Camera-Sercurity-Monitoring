"""Tests for `/cameras/{id}/video` — the H.264 WebSocket the live grid renders from.

These exercise the wire contract the browser depends on, because every violation of it is
silent in a tab: an unauthenticated socket that stays open, an `init` message whose codec
string does not match the bitstream, media bytes before the init segment, or a camera that
cannot encode leaving the client waiting instead of falling back to MJPEG.

Most of these stub the encoder: what is under test is the route's protocol and its auth
gate, and the encoder's own behaviour has its own module. The last one does not — it runs a
real camera through the real encoder and decodes what comes off the socket, which is the
only check that the bytes a browser receives are actually playable video.
"""

import io
import threading
import time
from collections import deque

import pytest
from starlette.websockets import WebSocketDisconnect

from src.models.camera_schemas import CameraInfo


def register_camera(client, camera_id: str, name: str) -> CameraInfo:
    """Put a camera straight into the service registry.

    Not via `POST /cameras`: the route under test only looks the camera up by id, and
    creating one for real would start decode threads for a test about socket framing.
    """
    camera = CameraInfo(
        id=camera_id,
        name=name,
        source_type="demo",
        source="clip.mp4",
        resolved_source="clip.mp4",
        intrusion_enabled=False,
        target_fps=5.0,
        stream_url=f"/api/v1/cameras/{camera_id}/stream",
        video_url=f"/api/v1/cameras/{camera_id}/video",
    )
    client.app.state.monitoring._cameras[camera_id] = camera
    return camera


class StubEncoder:
    """Minimal stand-in for `H264StreamEncoder` with the surface the route touches."""

    def __init__(self, *, available=True, init_segment=b"INIT", codec_name="libx264"):
        self._available = available
        self._init = init_segment
        self.codec_name = codec_name
        self.subscriptions = []
        self.unsubscribed = []

    @property
    def available(self):
        return self._available

    def init_segment(self):
        return self._init

    def subscribe(self):
        entry = (deque(), threading.Event(), {"waiting_for_keyframe": False, "dropped": 0, "closed": False})
        self.subscriptions.append(entry)
        return entry

    def unsubscribe(self, entry):
        self.unsubscribed.append(entry)

    def push(self, *fragments):
        """Deliver fragments to every subscriber, as the display loop's encode() would."""
        for queue, ready, _ in self.subscriptions:
            queue.extend(fragments)
            ready.set()

    def end(self):
        """Stop the stream, as a camera reaching EOF or a failing encoder would."""
        self._available = False
        for _, ready, state in self.subscriptions:
            state["closed"] = True
            ready.set()


@pytest.fixture
def camera_with_encoder(guard_client, monkeypatch):
    """Register one camera whose H.264 encoder is a stub under our control.

    The camera is injected straight into the service rather than created through the API:
    the route only looks it up by id, and going through `POST /cameras` would start real
    decode threads for a test about the socket's protocol.
    """
    service = guard_client.app.state.monitoring
    register_camera(guard_client, "cam_ws", "Socket Camera")

    encoder = StubEncoder()
    monkeypatch.setattr(service, "h264_encoder", lambda camera_id: encoder if camera_id == "cam_ws" else None)
    return guard_client, encoder


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_socket_is_closed_for_a_signed_out_viewer(anon_client):
    """The MJPEG route this replaces is behind `require_guard`; the socket must match it.

    A WebSocket cannot carry a 401, so the gate shows up as a close with a policy code —
    and the connection must never reach the point of sending video.

    The camera is registered first and the close reason is asserted, not just the code: an
    unregistered camera is *also* refused with 1008, so without both this test would pass
    against a route with no auth check at all.
    """
    register_camera(anon_client, "cam_ws", "Socket Camera")

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with anon_client.websocket_connect("/api/v1/cameras/cam_ws/video") as socket:
            socket.receive()
    assert excinfo.value.code == 1008
    assert "authentication" in excinfo.value.reason.lower()


def test_socket_is_closed_for_a_manager(manager_client):
    """Watching footage is the guard's surface. A manager is authenticated but not allowed,
    and must be refused rather than handed a feed."""
    register_camera(manager_client, "cam_ws", "Socket Camera")

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with manager_client.websocket_connect("/api/v1/cameras/cam_ws/video") as socket:
            socket.receive()
    assert excinfo.value.code == 1008
    assert "authentication" in excinfo.value.reason.lower()


def test_unknown_camera_is_refused(guard_client):
    """Refused for the camera, not for the session — an authenticated guard reaches this."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with guard_client.websocket_connect("/api/v1/cameras/ghost/video") as socket:
            socket.receive()
    assert excinfo.value.code == 1008
    assert "ghost" in excinfo.value.reason


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------


def test_init_message_precedes_the_init_segment_and_then_the_fragments(camera_with_encoder):
    """MSE requires exactly this order, and the client parses it positionally.

    The codec string is asserted against the encoder's actual profile: MSE silently refuses
    a source buffer whose codec does not describe the bitstream, so this pins the two
    together — `avc1.42E01F` is Baseline (0x42) level 3.1 (0x1F), matching `_codec_options`.
    """
    client, encoder = camera_with_encoder

    with client.websocket_connect("/api/v1/cameras/cam_ws/video") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "init"
        assert hello["codec"] == "libx264"
        assert hello["mime"] == 'video/mp4; codecs="avc1.42E01F"'

        assert socket.receive_bytes() == b"INIT"

        encoder.push(b"FRAG-1", b"FRAG-2")
        assert socket.receive_bytes() == b"FRAG-1"
        assert socket.receive_bytes() == b"FRAG-2"


def test_unavailable_encoder_tells_the_client_to_fall_back(guard_client, monkeypatch):
    """The server saying so explicitly is what keeps a client off a socket that will never
    produce video — without it the tile shows nothing rather than switching to MJPEG."""
    service = guard_client.app.state.monitoring
    register_camera(guard_client, "cam_dead", "No Encoder")
    monkeypatch.setattr(service, "h264_encoder", lambda camera_id: StubEncoder(available=False))

    with guard_client.websocket_connect("/api/v1/cameras/cam_dead/video") as socket:
        message = socket.receive_json()

    assert message["type"] == "fallback"
    assert message["reason"]


def test_camera_without_h264_at_all_tells_the_client_to_fall_back(guard_client, monkeypatch):
    """`h264_encoder` returns None when the camera has no worker, or H.264 is off for it."""
    service = guard_client.app.state.monitoring
    register_camera(guard_client, "cam_none", "No H264")
    monkeypatch.setattr(service, "h264_encoder", lambda camera_id: None)

    with guard_client.websocket_connect("/api/v1/cameras/cam_none/video") as socket:
        assert socket.receive_json()["type"] == "fallback"


def test_stream_ending_is_announced_rather_than_dropped(camera_with_encoder):
    """A camera that stops mid-session must produce a `fallback`, not silence.

    Silence is indistinguishable from a stalled feed at the client, which would leave the
    tile on a frozen frame instead of switching to the MJPEG image.
    """
    client, encoder = camera_with_encoder

    with client.websocket_connect("/api/v1/cameras/cam_ws/video") as socket:
        socket.receive_json()
        socket.receive_bytes()
        encoder.push(b"FRAG-1")
        assert socket.receive_bytes() == b"FRAG-1"

        encoder.end()
        assert socket.receive_json()["type"] == "fallback"


def test_idle_camera_is_dropped_close_to_the_timeout_not_late(camera_with_encoder, monkeypatch):
    """The send loop polls the subscriber's queue instead of blocking on a threading.Event
    (see the comment above `poll_interval` in the route), backing off while the camera stays
    quiet so it does not wake this coroutine every tick for the whole idle window. The backoff
    is capped, and this is what proves the cap actually bounds it: a camera producing nothing
    must still be noticed within a small multiple of the poll cap past the idle deadline, not
    left hanging because the interval grew unbounded.
    """
    import src.api.vision_routes as routes

    # A window long enough that an *uncapped* doubling backoff (0.02, 0.04, 0.08, ... with
    # no ceiling) would overshoot it by more than a second — short enough to keep the test
    # fast. Regression-tested: with the cap removed this assertion fails at ~2.7s elapsed.
    monkeypatch.setattr(routes, "_H264_IDLE_TIMEOUT_SEC", 2.0)
    monkeypatch.setattr(routes, "_H264_POLL_INTERVAL_MIN_SEC", 0.02)
    monkeypatch.setattr(routes, "_H264_POLL_INTERVAL_MAX_SEC", 0.1)

    client, encoder = camera_with_encoder
    started = time.monotonic()
    with client.websocket_connect("/api/v1/cameras/cam_ws/video") as socket:
        socket.receive_json()
        socket.receive_bytes()
        # No push(): the camera producing nothing after the init segment is exactly the
        # "gone quiet" case idle_deadline exists to catch.
        assert socket.receive_json()["type"] == "fallback"
    elapsed = time.monotonic() - started

    # Generous slack above the idle window plus one poll interval: enough to absorb test
    # scheduling jitter, tight enough that an unbounded or mis-capped backoff would fail it.
    assert elapsed < 2.0 + 0.1 + 0.5


def test_disconnecting_unsubscribes_from_the_encoder(camera_with_encoder):
    """A leaked subscriber keeps receiving fragments into a queue nobody drains, which is
    exactly the unbounded-growth case the encoder's queue cap exists to bound."""
    client, encoder = camera_with_encoder

    with client.websocket_connect("/api/v1/cameras/cam_ws/video") as socket:
        socket.receive_json()
        socket.receive_bytes()

    assert len(encoder.unsubscribed) == 1
    assert encoder.unsubscribed[0] is encoder.subscriptions[0]


def test_no_media_is_sent_before_the_init_segment_exists(guard_client, monkeypatch):
    """A camera that has not produced a frame yet has no init segment. Sending fragments
    without one puts MSE into an error state that never recovers, so the route must wait —
    and give up into a fallback rather than wait forever."""
    import src.api.vision_routes as routes

    service = guard_client.app.state.monitoring
    register_camera(guard_client, "cam_slow", "Slow Start")
    encoder = StubEncoder(init_segment=None)
    monkeypatch.setattr(service, "h264_encoder", lambda camera_id: encoder)
    # The real timeout is sized for an RTSP handshake; the behaviour under test is what
    # happens when it expires, so it is shortened rather than waited out.
    monkeypatch.setattr(routes, "_H264_START_TIMEOUT_SEC", 0.3)

    with guard_client.websocket_connect("/api/v1/cameras/cam_slow/video") as socket:
        message = socket.receive_json()

    assert message["type"] == "fallback"


def test_a_camera_that_starts_late_still_gets_a_stream(guard_client, monkeypatch):
    """The counterpart to the timeout: an encoder whose first frame lands during the wait
    must be picked up, not written off."""
    service = guard_client.app.state.monitoring
    register_camera(guard_client, "cam_late", "Late Start")

    encoder = StubEncoder(init_segment=None)
    calls = {"n": 0}

    def init_segment():
        calls["n"] += 1
        # None on the first poll, then ready — the camera finished connecting.
        return b"LATE-INIT" if calls["n"] > 1 else None

    encoder.init_segment = init_segment
    monkeypatch.setattr(service, "h264_encoder", lambda camera_id: encoder)

    with guard_client.websocket_connect("/api/v1/cameras/cam_late/video") as socket:
        assert socket.receive_json()["type"] == "init"
        assert socket.receive_bytes() == b"LATE-INIT"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_socket_delivers_decodable_video_from_a_real_camera(guard_client):
    """The end-to-end claim, asserted rather than eyeballed: bytes leaving the WebSocket
    decode back into the right number of frames at the right size."""
    av = pytest.importorskip("av", reason="decoding the socket's output needs PyAV")

    demos = guard_client.get("/api/v1/demo-videos").json()
    if not demos:
        pytest.skip("no demo clips bundled in this checkout")
    response = guard_client.post(
        "/api/v1/cameras",
        json={"name": "E2E", "source_type": "demo", "source": demos[0]["id"], "loop_source": True},
    )
    assert response.status_code in (200, 201), response.text
    camera = response.json()
    assert camera["video_url"] == f"/api/v1/cameras/{camera['id']}/video"

    try:
        time.sleep(3)  # let the worker open the clip and encode a first GOP
        with guard_client.websocket_connect(camera["video_url"]) as socket:
            hello = socket.receive_json()
            assert hello["type"] == "init", hello
            data = socket.receive_bytes()
            for _ in range(40):
                data += socket.receive_bytes()

        frames = list(av.open(io.BytesIO(data)).decode(video=0))
        assert len(frames) > 20, f"only {len(frames)} frames decoded from {len(data)} bytes"
        assert frames[0].height == 480, "display path should downscale to 480p"
        print(f"\n  {len(data)} bytes -> {len(frames)} frames at {frames[0].width}x{frames[0].height}")
    finally:
        guard_client.delete(f"/api/v1/cameras/{camera['id']}")

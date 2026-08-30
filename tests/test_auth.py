"""Auth, roles, and what each one is allowed to do.

None of these load YOLO weights or reach the network: `conftest.auth_app` patches in the null
detector and hashes at `TEST_WORK_FACTOR`, so the whole file runs in about a second.

The tests are ordered by what they protect, most consequential first — privilege escalation,
then the 401 boundary, then the session lifecycle, then attribution.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from src.api.vision_routes import event_stream
from src.services.login_limiter import LoginLimiter
from src.services.security import hash_password, verify_password
from src.services.sessions import COOKIE_NAME, SessionStore
from src.services.user_store import User, UserStore
from tests.conftest import TEST_WORK_FACTOR

# Every route a guard must not reach, as (method, path). Kept as a list, and swept in a loop,
# even though the statistics report is currently the only entry: a second manager-only route
# added later is one line here rather than a new test nobody writes.
MANAGER_ONLY = [
    ("get", "/api/v1/statistics"),
]

# Guard-only product operations. Shared system status is listed separately below.
AUTHENTICATED_ONLY = [
    ("get", "/api/v1/cameras"),
    ("get", "/api/v1/alerts"),
    ("post", "/api/v1/alerts/anything/acknowledge"),
    ("post", "/api/v1/cameras"),
    ("delete", "/api/v1/cameras/anything"),
    ("get", "/api/v1/demo-videos"),
    ("post", "/api/v1/uploads"),
    ("get", "/api/v1/preview-frame?source_type=demo&source=x.mp4"),
]

OPERATOR_ONLY = [
    ("get", "/api/v1/monitoring/status"),
    ("get", "/api/v1/incidents"),
    ("get", "/api/v1/evidence/nope"),
    ("get", "/api/v1/alerts/nope/similar"),
]


def _present(client, value: str) -> None:
    """Put `value` in the session cookie, replacing whatever the client already holds.

    The domain is explicit because `jar.set()` without one stores a *domainless* cookie that
    coexists with the server's `testserver.local` one rather than replacing it — after which
    every read raises `CookieConflict` and every request sends both. The per-request
    `cookies={...}` argument is the other route to the same place and is deprecated.
    """
    client.cookies.set(COOKIE_NAME, value, domain="testserver.local", path="/")


class _StreamRequest:
    """The three things `event_stream` asks of a `Request`, and nothing else.

    Enough to drive the route's generator on the test's own event loop. `is_disconnected`
    always answers False, so the only thing that can end the stream is the revocation check
    under test — a real client disconnect would mask exactly the bug these tests look for.
    """

    def __init__(self, client, username: str) -> None:
        self.app = client.app
        self.cookies = {COOKIE_NAME: client.cookies.get(COOKIE_NAME)}
        assert self.cookies[COOKIE_NAME], f"no session cookie for {username}"

    async def is_disconnected(self) -> bool:
        return False


def _open_stream(client, username: str):
    """Call the route the way FastAPI would, minus the framework.

    `event_stream` takes its identity from the router's `require_guard` dependency rather
    than re-resolving the session itself, so calling the coroutine directly has to supply
    what the injector otherwise would. The cookie on `_StreamRequest` is still what the
    in-loop revocation check reads — that is the thing under test and it is untouched.
    """
    return event_stream(
        _StreamRequest(client, username),
        guard=User(username=username, role="guard", display_name=username),
    )


# ---------------------------------------------------------------- privilege boundary
# The two sweeps below loop rather than parametrize, and collect every failure before
# asserting. Each case needs a fresh app (isolated roster, sessions and registry), and
# standing one up costs ~0.5s — mostly `MonitoringService.__init__`. Parametrizing the eleven
# cases would add several seconds to a suite whose whole point is that you can just run it.
# Collecting the failures keeps what parametrize is actually worth here: one run tells you
# *every* route that is wrong, not just the first.
def test_a_guard_cannot_reach_a_manager_route(guard_client):
    """The escalation boundary. A guard is signed in, so these must be 403 and not 401.

    The statistics report is the only manager-gated surface. Camera CRUD, uploads and the
    RTSP preview are deliberately *not* in here — the guard is the one who configures
    cameras, so those live on the shared gate and `test_a_guard_can_configure_cameras`
    asserts the opposite property about them.
    """
    wrong = []
    for method, path in MANAGER_ONLY:
        status = getattr(guard_client, method)(path).status_code
        if status != 403:
            wrong.append(f"{method.upper()} {path} -> {status}")
    assert not wrong, f"reachable by a guard: {wrong}"


def test_a_manager_can_reach_the_manager_routes(manager_client):
    """The other half: the gate must not be refusing everyone."""
    assert manager_client.get("/api/v1/statistics").status_code == 200


def test_a_guard_can_configure_cameras(guard_client):
    """Camera setup is the guard's, and only a test can tell that from the router it sits on.

    Each assertion is a status that proves the *gate* passed and the handler ran — a 404 for
    a camera that does not exist, a 400 for a demo file that does not exist. A 403 anywhere
    here means the route drifted back onto the manager router, which would leave the person
    who runs the room unable to add the camera they are meant to watch.
    """
    assert guard_client.get("/api/v1/demo-videos").status_code == 200
    assert guard_client.delete("/api/v1/cameras/nope").status_code == 404
    response = guard_client.post(
        "/api/v1/cameras",
        json={"name": "Lobby", "source_type": "demo", "source": "does_not_exist.mp4"},
    )
    assert response.status_code == 400, response.text


def test_a_manager_cannot_reach_guard_routes_but_can_read_shared_status(manager_client):
    """The manager boundary is *read versus act*, and this is where that line falls.

    A manager reviewing the room's work reaches the durable incident history and the stored
    evidence stills it links to — auditing a guard's verdict against a label and a timestamp
    alone is not auditing it. What stays refused is the live surface and every write: the
    camera list, the live alert buffer behind the guard's rail, and the verdict route below.
    """
    # Refused: live footage and camera control.
    assert manager_client.get("/api/v1/cameras").status_code == 403
    # Refused: the live ring buffer behind the guard's rail. The manager's equivalent is
    # `/incidents`, which is the durable log rather than this process's tail.
    assert manager_client.get("/api/v1/alerts").status_code == 403
    # Refused: the guard's write path. Reviewing is not judging.
    assert manager_client.post("/api/v1/alerts/anything/verdict", json={"verdict": "real"}).status_code == 403
    assert manager_client.post("/api/v1/alerts/anything/acknowledge").status_code == 403
    # Refused: the live alert stream.
    assert manager_client.get("/api/v1/events").status_code == 403

    # Allowed: shared status, and the read-only history surface.
    assert manager_client.get("/api/v1/monitoring/status").status_code == 200
    assert manager_client.get("/api/v1/incidents").status_code == 200
    assert manager_client.get("/api/v1/alerts/anything/similar").status_code == 200
    # 404 rather than 403: the gate passed and the handler ran on an id that does not exist.
    # A 403 here would mean evidence had drifted back behind the guard-only router and the
    # thumbnails in the history list would silently stop resolving.
    assert manager_client.get("/api/v1/evidence/does-not-exist").status_code == 404


def test_a_guard_keeps_the_history_surface_too(guard_client):
    """Widening these to the shared gate must not have moved them off the guard.

    The guard's rail and modal read `/evidence` and `/similar` on every alert they open. A
    change that handed those to the manager by taking them from the guard would break the
    screen the product exists for, and would do it silently — the images would just stop
    appearing.
    """
    assert guard_client.get("/api/v1/incidents").status_code == 200
    assert guard_client.get("/api/v1/alerts/anything/similar").status_code == 200
    assert guard_client.get("/api/v1/evidence/does-not-exist").status_code == 404
    assert guard_client.get("/api/v1/alerts").status_code == 200


def test_every_gated_route_is_401_when_signed_out(anon_client):
    """No route on either router answers without a session.

    Swept over both lists rather than spot-checked: the whole reason the gates are
    router-level is that a route is protected by *where it was added*, and this is the test
    that would catch someone mounting a new one on a bare `APIRouter`.
    """
    wrong = []
    for method, path in MANAGER_ONLY + AUTHENTICATED_ONLY + OPERATOR_ONLY:
        status = getattr(anon_client, method)(path).status_code
        if status != 401:
            wrong.append(f"{method.upper()} {path} -> {status}")
    assert not wrong, f"answered without a session: {wrong}"


def test_health_stays_public(anon_client):
    """It is the Docker healthcheck, which carries no cookie. Gating it fails the container."""
    assert anon_client.get("/health").status_code == 200


# ---------------------------------------------------------------- session integrity
def test_a_tampered_or_unknown_session_is_401_not_500(anon_client):
    """Garbage in the cookie must fail closed, quietly.

    A 500 here would be worse than an auth bug on its own: it says the value reached code
    that did not expect it, and it is the signal an attacker uses to find what does.
    """
    for value in ("", "not-a-session", "../../etc/passwd", "a" * 5000):
        _present(anon_client, value)
        response = anon_client.get("/api/v1/cameras")
        assert response.status_code == 401, f"cookie {value!r} gave {response.status_code}"


def test_logout_revokes_the_session_server_side(guard_client):
    """The reason sessions are stored at all.

    Replaying the cookie value after logout must fail. A signed cookie carrying the username
    would pass this replay — it stays valid until it expires, and nothing on the server can
    end it. That is the trade this store exists to avoid on a system streaming live footage.
    """
    stolen = guard_client.cookies.get(COOKIE_NAME)
    assert stolen

    assert guard_client.post("/api/v1/auth/logout").status_code == 204
    assert guard_client.get("/api/v1/cameras").status_code == 401

    # Replayed as an attacker would, having captured the value while it was live.
    _present(guard_client, stolen)
    assert guard_client.get("/api/v1/cameras").status_code == 401


def test_login_replaces_any_pre_existing_session(anon_client):
    """Session fixation: the id a browser arrives holding must not survive login.

    Asserted server-side rather than by comparing cookie strings. A new id in the response
    would satisfy the letter of "rotated" while the planted session stayed resolvable — and
    that surviving session, whose id the attacker already knows, is the entire attack.
    """
    first = anon_client.post("/api/v1/auth/login", json={"username": "gwen", "password": "guard-password"})
    planted = first.cookies.get(COOKIE_NAME)
    assert planted

    _present(anon_client, planted)
    second = anon_client.post("/api/v1/auth/login", json={"username": "gwen", "password": "guard-password"})
    issued = second.cookies.get(COOKIE_NAME)

    assert second.status_code == 200
    assert issued and issued != planted
    # The one the browser arrived with is gone, not merely superseded.
    _present(anon_client, planted)
    assert anon_client.get("/api/v1/cameras").status_code == 401
    _present(anon_client, issued)
    assert anon_client.get("/api/v1/cameras").status_code == 200


def test_a_wrong_password_and_an_unknown_user_are_indistinguishable(anon_client):
    """Same status, same message. Anything else is a free username oracle."""
    wrong = anon_client.post("/api/v1/auth/login", json={"username": "gwen", "password": "nope"})
    unknown = anon_client.post("/api/v1/auth/login", json={"username": "nobody", "password": "nope"})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_repeated_failed_logins_are_temporarily_limited(anon_client):
    anon_client.app.state.login_limiter = LoginLimiter(
        max_failures=3,
        window_seconds=60,
        lockout_seconds=30,
    )

    for _ in range(2):
        response = anon_client.post(
            "/api/v1/auth/login",
            json={"username": "gwen", "password": "wrong"},
        )
        assert response.status_code == 401

    blocked = anon_client.post(
        "/api/v1/auth/login",
        json={"username": "gwen", "password": "wrong"},
    )
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0

    still_blocked = anon_client.post(
        "/api/v1/auth/login",
        json={"username": "gwen", "password": "guard-password"},
    )
    assert still_blocked.status_code == 429


def test_the_password_hash_never_leaves_the_server(anon_client):
    """`User` and `UserRecord` are two types precisely so this cannot regress."""
    body = anon_client.post("/api/v1/auth/login", json={"username": "morgan", "password": "manager-password"}).json()

    assert body == {"username": "morgan", "role": "manager", "display_name": "Morgan the Manager"}
    assert "password_hash" not in body
    assert "password_hash" not in anon_client.get("/api/v1/auth/me").json()


def test_disabling_an_account_ends_its_live_session(guard_client, auth_app):
    """The role and enabled flag are re-read per request, not trusted from the session.

    Without that, disabling a guard mid-shift would leave them working for up to 12 hours —
    the absolute TTL — with a session that says they are still staff.
    """
    _, users = auth_app
    assert guard_client.get("/api/v1/cameras").status_code == 200

    users._users["gwen"].active = False
    assert guard_client.get("/api/v1/cameras").status_code == 401


def test_a_demoted_manager_loses_manager_routes_immediately(manager_client, auth_app):
    """Same mechanism, the other direction: the session stores no role to go stale."""
    _, users = auth_app
    assert manager_client.get("/api/v1/statistics").status_code == 200

    users._users["morgan"].role = "guard"
    assert manager_client.get("/api/v1/statistics").status_code == 403


# ---------------------------------------------------------------- session store unit
def test_an_idle_session_expires_and_a_stream_poll_does_not_keep_it_alive(tmp_path):
    """Both halves of `resolve`'s contract, which only make sense together.

    `touch=False` exists for the streaming routes: an MJPEG response left open on a wall
    display would otherwise refresh `last_seen` forever and defeat the idle timeout that is
    there precisely for an unattended terminal.
    """
    store = SessionStore(path=str(tmp_path / "s.json"), idle_minutes=30.0)
    session_id = store.create("gwen")

    # Age it past the idle window by hand — sleeping 30 minutes is not a test.
    store._sessions[session_id].last_seen = datetime.now(UTC) - timedelta(minutes=31)
    assert store.resolve(session_id, touch=False) is None

    fresh = store.create("gwen")
    store._sessions[fresh].last_seen = datetime.now(UTC) - timedelta(minutes=29)
    assert store.resolve(fresh, touch=False) is not None
    # Still 29 minutes old: the poll read it without refreshing it.
    assert datetime.now(UTC) - store._sessions[fresh].last_seen > timedelta(minutes=28)


def test_the_absolute_ttl_expires_a_session_that_is_still_being_used(tmp_path):
    """A busy tab must not be able to hold a session open past the shift-length ceiling."""
    store = SessionStore(path=str(tmp_path / "s.json"), ttl_hours=12.0, idle_minutes=30.0)
    session_id = store.create("gwen")

    store._sessions[session_id].created_at = datetime.now(UTC) - timedelta(hours=13)
    # `last_seen` is now, so only the absolute TTL can end this one.
    assert store.resolve(session_id) is None


def test_expired_sessions_are_not_resurrected_by_a_restart(tmp_path):
    """The snapshot must not undo an expiry that elapsed while the process was down."""
    path = str(tmp_path / "s.json")
    store = SessionStore(path=path, idle_minutes=30.0)
    stale = store.create("gwen")
    live = store.create("morgan")
    store._sessions[stale].last_seen = datetime.now(UTC) - timedelta(minutes=31)
    store._persist_locked()

    restarted = SessionStore(path=path, idle_minutes=30.0)
    assert restarted.resolve(stale) is None
    assert restarted.resolve(live) is not None


def test_the_session_cap_evicts_the_least_recently_seen(tmp_path):
    """Otherwise every abandoned tab is a row nothing removes until its TTL runs out."""
    store = SessionStore(path=str(tmp_path / "s.json"), max_sessions=3)
    ids = [store.create(f"user{i}") for i in range(3)]
    store._sessions[ids[0]].last_seen = datetime.now(UTC) - timedelta(minutes=5)

    store.create("newcomer")
    assert store.resolve(ids[0]) is None
    assert store.active_count() == 3


# ---------------------------------------------------------------- password records
def test_a_malformed_password_record_fails_closed(tmp_path):
    """Fails closed on every shape a hand-edited or truncated file can produce.

    The alternative is a 500 on the login route, which both leaks that the account exists and
    takes the app down for one corrupt row rather than denying one login.
    """
    for record in ("", "garbage", "scrypt$n=bad$c2FsdA==$aGFzaA==", "argon2$n=1$x$y", "scrypt$n=3,r=8,p=1$x$y"):
        assert verify_password("anything", record) is False


def test_a_hash_verifies_at_the_work_factor_it_was_made_with():
    """Parameters live inside the record, so raising the default cannot lock anyone out."""
    stored = hash_password("correct horse", work_factor=TEST_WORK_FACTOR)

    assert verify_password("correct horse", stored) is True
    assert verify_password("Correct horse", stored) is False
    assert f"n={2**TEST_WORK_FACTOR}," in stored


def test_two_hashes_of_the_same_password_differ(tmp_path):
    """A fresh salt per hash — otherwise the file tells you which accounts share a password."""
    a = hash_password("same", work_factor=TEST_WORK_FACTOR)
    b = hash_password("same", work_factor=TEST_WORK_FACTOR)
    assert a != b


def test_usernames_are_matched_case_insensitively_but_stored_as_typed(tmp_path):
    """A guard typing "Gwen" at 3am should not be told their password is wrong."""
    store = UserStore(path=str(tmp_path / "u.json"))
    store.add_user("Gwen", "pw", "guard", work_factor=TEST_WORK_FACTOR)

    assert store.authenticate("gwen", "pw") is not None
    assert store.authenticate("  GWEN  ", "pw") is not None
    assert store.list_users()[0].username == "Gwen"


def test_a_corrupt_roster_denies_every_login_without_crashing(tmp_path):
    """A malformed file must not stop the app booting — the alert pipeline matters more."""
    path = tmp_path / "u.json"
    path.write_text("{not json", encoding="utf-8")

    store = UserStore(path=str(path))
    assert store.list_users() == []
    assert store.authenticate("gwen", "pw") is None


# ---------------------------------------------------------------- the SSE reconnect guard
def test_a_dead_session_never_opens_a_stream(anon_client):
    """The first half of the reconnect-loop fix: auth is checked before the stream exists.

    `EventSource` retries on its own, forever. If a dead session got a 200 and an
    immediately-closing stream, the browser would reconnect on a loop and turn one expired
    session into a permanent request storm. A plain 401 is what makes the client give up.
    """
    assert anon_client.get("/api/v1/events").status_code == 401


@pytest.mark.asyncio
async def test_a_revoked_session_ends_the_event_stream(auth_app):
    """A session revoked *mid-stream* must close the response, not just fail the next request.

    Authenticating once at connect is not enough for a response that stays open for hours:
    without the in-loop check, logging out would leave the alert feed running until the
    browser closed it, and "logout ends the session" would be a client-side courtesy rather
    than something the server enforces.

    Driven against the route's async generator directly. Both in-process HTTP clients
    (`TestClient` and `httpx.ASGITransport`) buffer the entire response body before returning,
    so neither can observe a stream that has not finished — a test written through either one
    hangs forever instead of failing. Verified separately against a real uvicorn socket, where
    the stream closes ~0ms after the revocation is published.
    """
    test_client, _ = auth_app
    test_client.post("/api/v1/auth/login", json={"username": "gwen", "password": "guard-password"})
    service = test_client.app.state.monitoring
    service.bind_loop(asyncio.get_running_loop())

    response = await _open_stream(test_client, "gwen")
    events = response.body_iterator

    service.publish_event("alert.created", {"id": "warmup"})
    assert "alert.created" in await asyncio.wait_for(anext(events), timeout=5.0)

    service.publish_event("auth.revoked", {"username": "gwen"})
    assert "auth.revoked" in await asyncio.wait_for(anext(events), timeout=5.0)

    # The generator is exhausted, which is the server closing the connection. Anything else
    # means the browser is left holding an open stream on a session that no longer exists.
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(events), timeout=5.0)


@pytest.mark.asyncio
async def test_a_revocation_for_someone_else_leaves_the_stream_open(auth_app):
    """`_publish` has no per-user index, so every stream sees every revocation.

    Filtering is by username on the way out. Without it, one guard signing out at the end of
    a shift would sign out everyone still watching — on a 24/7 monitoring wall, that is the
    whole control room going dark because one person left.
    """
    test_client, _ = auth_app
    test_client.post("/api/v1/auth/login", json={"username": "gwen", "password": "guard-password"})
    service = test_client.app.state.monitoring
    service.bind_loop(asyncio.get_running_loop())

    response = await _open_stream(test_client, "gwen")
    events = response.body_iterator

    service.publish_event("auth.revoked", {"username": "someone-else"})
    service.publish_event("alert.created", {"id": "still-here"})

    # The revocation is skipped silently and the next real event arrives, so this guard's
    # stream is still running.
    delivered = await asyncio.wait_for(anext(events), timeout=5.0)
    assert "still-here" in delivered
    assert "auth.revoked" not in delivered

    await events.aclose()


# ---------------------------------------------------------------- attribution
def test_the_verdict_records_which_guard_judged_it(guard_client):
    """`judged_by` is what makes the labelled dataset auditable when two guards disagree."""
    import numpy as np

    service = guard_client.app.state.monitoring
    service._emit_alert(
        camera_id="cam_1",
        label="fall_detection",
        confidence=0.9,
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
    )
    alert_id = service.list_alerts()[0].id

    assert guard_client.post(f"/api/v1/alerts/{alert_id}/verdict", json={"verdict": "real"}).status_code == 204

    judged = next(a for a in service.list_alerts() if a.id == alert_id)
    assert judged.operator_verdict == "real"
    assert judged.judged_by == "gwen"


def test_acknowledge_still_records_no_judgement(guard_client):
    """Dismissing a modal is not a verdict, and must not enter the precision dataset.

    Pinned here as well as in the eval-report tests because auth added a second writer to
    that file: it would be easy to start stamping `judged_by` on an acknowledge and turn
    every dismissal into an unlabelled row counted as neither a hit nor a miss.
    """
    import numpy as np

    service = guard_client.app.state.monitoring
    service._emit_alert(
        camera_id="cam_1", label="intrusion", confidence=0.5, frame=np.zeros((120, 160, 3), dtype=np.uint8)
    )
    alert_id = service.list_alerts()[0].id

    assert guard_client.post(f"/api/v1/alerts/{alert_id}/acknowledge").status_code == 204

    acked = next(a for a in service.list_alerts() if a.id == alert_id)
    assert acked.acknowledged is True
    assert acked.operator_verdict is None
    assert acked.judged_by is None


# ---------------------------------------------------------------- statistics shape
def test_statistics_keeps_the_nulls_that_carry_meaning(manager_client):
    """Two values the response must not tidy away.

    `precision` is `None` — not `0.0` — when nothing has been reviewed, because 0% reads as
    "every alert was wrong" on a camera nobody has looked at. And `recall_note` is a
    `@property`, which Pydantic drops from `model_dump()`; it carries the report's central
    decision, so it is re-attached by hand and this is what proves it survived.
    """
    body = manager_client.get("/api/v1/statistics").json()

    assert "report" in body and "operational" in body
    assert "recall_note" in body["report"]
    assert "not computable" in body["report"]["recall_note"].lower()
    # Present-and-null, not absent: `exclude_none` here would erase the distinction.
    assert "rule_precision" in body["report"]
    assert body["report"]["rule_precision"] is None
    assert body["operational"]["total_alerts"] == 0

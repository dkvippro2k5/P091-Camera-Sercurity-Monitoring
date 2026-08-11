"""Auth helpers, called imperatively from handlers.

Deliberately plain functions taking `Request` rather than `Depends`-injected parameters,
mirroring `get_service(request)` in `vision_routes.py`. Every handler in that module already
takes `request: Request`, so identity is available anywhere without touching a signature —
and a reader can see where the check happens instead of inferring it from a default argument.

They are used two ways, which is not redundancy:

- As router-level `dependencies=[Depends(require_guard)]`, which gates *every* route on that
  router including ones nobody has written yet. That is the property worth having: a new
  endpoint is protected because of where it was added, not because its author remembered.
- Called inside a handler that needs to know *who* is asking — the verdict route stamps the
  guard's username onto the history row, and the SSE route needs its own username to decide
  whether an `auth.revoked` broadcast is about it.
"""

from fastapi import HTTPException, Request

from src.services.sessions import COOKIE_NAME, SessionStore
from src.services.user_store import User, UserStore

# 401 tells the browser "log in"; 403 tells it "you are logged in but this is not yours".
# The frontend branches on exactly this difference: 401 returns to the login screen, 403
# surfaces an error without throwing away the session.
_NOT_AUTHENTICATED = "Not signed in"
_NOT_PERMITTED = "This action is for managers only"


def _stores(request: Request) -> tuple[UserStore, SessionStore]:
    """Pull the auth singletons off app state, as `get_service` does for monitoring."""
    users = getattr(request.app.state, "users", None)
    sessions = getattr(request.app.state, "sessions", None)
    if users is None or sessions is None:
        # 503 rather than 401: the caller's credentials are not the problem, and a 401 here
        # would send a perfectly valid operator back to a login screen that cannot work.
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    return users, sessions


def current_user(request: Request, touch: bool = True) -> User:
    """Resolve the session cookie to a user, or raise 401.

    The role is re-read from `UserStore` on every request rather than trusted from the
    session record, so disabling an account or changing its role takes effect on the next
    request instead of whenever a 12-hour session happens to end.
    """
    users, sessions = _stores(request)
    session_id = request.cookies.get(COOKIE_NAME)
    session = sessions.resolve(session_id, touch=touch)
    if session is None:
        raise HTTPException(status_code=401, detail=_NOT_AUTHENTICATED)

    user = users.get(session.username)
    if user is None:
        # The account was deleted or disabled while the session was live. Drop the session
        # too — leaving it resolvable would keep handing out 401s against a record that can
        # never succeed again.
        sessions.destroy(session_id)
        raise HTTPException(status_code=401, detail=_NOT_AUTHENTICATED)
    return user


def require_guard(request: Request) -> User:
    """Any signed-in operator. Guards and managers both pass.

    Named for the surface it protects rather than the role it demands: this is everything a
    guard does — the live grid, the alert queue, and camera configuration.

    A manager is not offered any of it in the UI, and the gate still admitting them is
    deliberate rather than an oversight. `/monitoring/status` is polled every 3 seconds by
    every signed-in browser to feed the masthead's camera vitals, and the frontend signs out
    on any 401 — so a role check here would evict a manager from their own session seconds
    after they logged in. The manager's restriction is enforced by not routing them to these
    screens, which means a direct API call still reaches them. Accepted, and the reason it is
    acceptable is that the two roles are colleagues in one control room, not adversaries.
    """
    return current_user(request)


def require_manager(request: Request) -> User:
    """Managers only — the statistics report, and nothing else."""
    user = current_user(request)
    if user.role != "manager":
        raise HTTPException(status_code=403, detail=_NOT_PERMITTED)
    return user


def session_is_live(request: Request) -> bool:
    """Is this request's session still valid? Never raises.

    For the long-lived streaming routes to poll. Authenticating once at connect is not enough
    for a response that stays open for hours: logging out would leave the MJPEG feed and the
    alert stream running until the browser closed them, which makes "logout ends the session"
    a client-side courtesy rather than something the server enforces.

    Deliberately does **not** touch the session. A stream that refreshed `last_seen` on every
    poll would keep an unattended terminal signed in forever, defeating the idle timeout that
    exists precisely for a control room with the grid left up on a screen.
    """
    try:
        users, sessions = _stores(request)
    except HTTPException:
        # Auth not configured — let the stream run rather than killing the video feed over a
        # deployment problem the operator cannot do anything about.
        return True

    session = sessions.resolve(request.cookies.get(COOKIE_NAME), touch=False)
    return session is not None and users.get(session.username) is not None

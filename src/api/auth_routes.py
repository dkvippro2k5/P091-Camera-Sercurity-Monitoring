"""Login, logout, and "who am I" — the only public routes besides `/health`.

Session id in an `HttpOnly` cookie rather than a token in a header, because the two things
this UI depends on cannot send headers: the camera grid is an `<img src>` MJPEG stream and
alerts arrive over `EventSource`. Neither has a header API. A cookie is the only credential
that reaches *every* route uniformly, and `HttpOnly` puts it out of reach of any script on
the page, which a token in `localStorage` never is.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.api.deps import current_user
from src.config import get_settings
from src.services.sessions import COOKIE_NAME
from src.services.user_store import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    # Bounded so a multi-megabyte body cannot be fed into scrypt as a denial-of-service.
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Attach the session cookie with the flags that matter.

    - `httponly` — no script can read it, so an XSS bug cannot exfiltrate the session.
    - `samesite="lax"` — the browser will not attach it to cross-site POSTs, which is what
      stops another origin from submitting a verdict or deleting a camera on the operator's
      behalf. "lax" rather than "strict" so following a link into the dashboard still works.
    - `secure` — from config, and **off by default**: a browser silently drops a `Secure`
      cookie sent over plain HTTP, so forcing it on the default `http://localhost:8000` setup
      produces a login that appears to succeed and then bounces straight back, with no error
      to explain why. Turn it on when TLS actually terminates in front of the app.
    - `max_age` — matches the session's absolute TTL, so a closed browser does not leave a
      cookie that outlives the record it points at.
    """
    settings = get_settings()
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        max_age=int(settings.session_ttl_hours * 3600),
        path="/",
    )


@router.post("/auth/login", response_model=User)
async def login(payload: LoginRequest, request: Request, response: Response) -> User:
    """Exchange a username and password for a session cookie."""
    users = getattr(request.app.state, "users", None)
    sessions = getattr(request.app.state, "sessions", None)
    if users is None or sessions is None:
        raise HTTPException(status_code=503, detail="Authentication is not configured")

    limiter = getattr(request.app.state, "login_limiter", None)
    remote = request.client.host if request.client else "unknown"
    limit_key = f"{remote}:{payload.username.strip().lower()}"
    retry_after = limiter.retry_after(limit_key) if limiter else 0
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    # scrypt is deliberately expensive (~120 ms), which is the whole point of using it — but
    # that is 120 ms the event loop must not spend blocked, or a handful of concurrent logins
    # would stall every camera stream and SSE connection in the process.
    user = await asyncio.to_thread(users.authenticate, payload.username, payload.password)
    if user is None:
        retry_after = limiter.failed(limit_key) if limiter else 0
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="Too many failed login attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        # One message for every failure mode. Distinguishing "no such user" from "wrong
        # password" hands an attacker a free username oracle.
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    if limiter:
        limiter.succeeded(limit_key)

    # Any existing session is discarded rather than reused, so a session id an attacker
    # planted in the browser before login cannot survive into the authenticated session.
    # Same event-loop concern as `users.authenticate` above: on PostgresSessionStore this is
    # a real network round trip, not an in-memory dict operation.
    old_session_id = request.cookies.get(COOKIE_NAME)
    if old_session_id:
        await asyncio.to_thread(sessions.destroy, old_session_id)

    new_session_id = await asyncio.to_thread(sessions.create, user.username)
    _set_session_cookie(response, new_session_id)
    return user


@router.post("/auth/logout", status_code=204)
async def logout(request: Request, response: Response) -> Response:
    """End the session server-side, then clear the cookie.

    Order matters: deleting the server record first means that even if the response is lost
    on the way back, the session is already dead. Clearing the cookie first would leave a
    live session whose id the browser has thrown away — unusable by the operator and still
    valid to anyone who captured it.
    """
    session_id = request.cookies.get(COOKIE_NAME)
    sessions = getattr(request.app.state, "sessions", None)
    username = await asyncio.to_thread(sessions.destroy, session_id) if sessions else None

    if username:
        # Tell any stream this operator has open to stop. Without it, `EventSource` keeps
        # reconnecting with the dead cookie and turns a logout into an endless 401 loop.
        service = getattr(request.app.state, "monitoring", None)
        if service is not None:
            service.publish_event("auth.revoked", {"username": username})

    result = Response(status_code=204)
    # `delete_cookie` must be given the same path/samesite/secure the cookie was set with,
    # or the browser treats it as a different cookie and quietly keeps the original.
    settings = get_settings()
    result.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
    )
    return result


@router.get("/auth/me", response_model=User)
async def me(request: Request) -> User:
    """Who the current cookie belongs to. The frontend calls this on load to restore state."""
    # current_user is called imperatively here rather than via Depends, so FastAPI's
    # automatic threadpool wrapping for sync dependencies does not apply -- same blocking
    # concern as the session calls above.
    return await asyncio.to_thread(current_user, request)

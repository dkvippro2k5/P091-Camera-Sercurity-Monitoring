"""Server-side session store: the thing that makes logout actually mean something.

A signed cookie carrying the username would be simpler and needs no storage, but it cannot be
revoked — a logged-out cookie stays valid until it expires, and there is no way to end a
session from the server. On a system that streams live camera footage that is the wrong
trade, so sessions live here and the cookie carries nothing but an opaque random id.

Mirrors the lock + never-raise-on-`OSError` shape of `alert_history.py`: the store is asked to
do its work on the request path, and an unwritable disk must degrade rather than break login.

**Two expiries, whichever fires first.** An absolute TTL (`SESSION_TTL_HOURS`, default 12 —
one shift) bounds how long a stolen cookie is useful. An idle timeout
(`SESSION_IDLE_MINUTES`, default 30) ends a session left open on an unattended terminal in a
room the public can walk through. Neither alone covers both cases: a pure idle timeout keeps
a session alive forever if a tab keeps polling, and a pure absolute TTL leaves an abandoned
console logged in for the rest of the shift.
"""

import json
import logging
import os
import secrets
import threading
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SESSIONS_PATH = os.path.join(_PROJECT_ROOT, "data", "sessions.json")

COOKIE_NAME = "p091_session"

# 32 bytes from `secrets.token_urlsafe` is ~43 characters of URL-safe base64 and 256 bits of
# entropy — not guessable, and short enough to sit in a cookie without comment.
_SESSION_ID_BYTES = 32


class Session(BaseModel):
    """One logged-in browser. `username` is the only identity here.

    The role is deliberately **not** stored: it is looked up from `UserStore` on each request
    so that demoting a manager or disabling an account takes effect immediately rather than
    at the end of their shift.
    """

    username: str
    created_at: datetime
    last_seen: datetime


class SessionStore:
    """Live sessions, in memory, snapshotted to disk so a restart does not log everyone out.

    The snapshot is a convenience, not a source of truth: it is rewritten on every mutation
    and reloaded at startup with expired rows dropped. If it cannot be written, sessions still
    work for the life of the process — the failure mode is "a restart logs everyone out",
    which is survivable and logged.
    """

    def __init__(
        self,
        path: str = DEFAULT_SESSIONS_PATH,
        ttl_hours: float = 12.0,
        idle_minutes: float = 30.0,
        max_sessions: int = 200,
    ) -> None:
        self.path = path
        self.ttl = timedelta(hours=ttl_hours)
        self.idle = timedelta(minutes=idle_minutes)
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._load()

    # ------------------------------------------------------------------ lifecycle
    def create(self, username: str) -> str:
        """Start a session and return its id. The caller sets the cookie."""
        now = datetime.now(UTC)
        session_id = secrets.token_urlsafe(_SESSION_ID_BYTES)
        with self._lock:
            self._sessions[session_id] = Session(username=username, created_at=now, last_seen=now)
            self._evict_locked(now)
            self._persist_locked()
        logger.info(f"Session opened for '{username}'")
        return session_id

    def resolve(self, session_id: str | None, touch: bool = True) -> Session | None:
        """Look up a session, dropping it if either expiry has passed.

        `touch=False` is for the streaming routes. A permanently-open MJPEG or SSE response
        would otherwise refresh `last_seen` forever and defeat the idle timeout entirely — an
        unattended terminal with the grid up would never time out, which is exactly the case
        the idle timeout exists for.
        """
        if not session_id:
            return None

        now = datetime.now(UTC)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            if self._is_expired(session, now):
                del self._sessions[session_id]
                self._persist_locked()
                logger.info(f"Session expired for '{session.username}'")
                return None

            if touch:
                session.last_seen = now
                # Not persisted here on purpose: writing the file on every authenticated
                # request would turn a `last_seen` bump into disk I/O on the request path.
                # Losing the most recent touches to a crash only shortens a session.
            return session

    def destroy(self, session_id: str | None) -> str | None:
        """End one session. Returns the username it belonged to, or None if unknown.

        The username goes back to the caller so the logout route can announce the revocation
        to any open stream for that user.
        """
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                self._persist_locked()
        if session is None:
            return None
        logger.info(f"Session closed for '{session.username}'")
        return session.username

    def destroy_all_for(self, username: str) -> int:
        """End every session belonging to one account. Used when disabling a user."""
        with self._lock:
            doomed = [sid for sid, session in self._sessions.items() if session.username == username]
            for sid in doomed:
                del self._sessions[sid]
            if doomed:
                self._persist_locked()
        if doomed:
            logger.info(f"Revoked {len(doomed)} session(s) for '{username}'")
        return len(doomed)

    def active_count(self) -> int:
        now = datetime.now(UTC)
        with self._lock:
            return sum(1 for session in self._sessions.values() if not self._is_expired(session, now))

    # ------------------------------------------------------------------ internals
    def _is_expired(self, session: Session, now: datetime) -> bool:
        return now - session.created_at >= self.ttl or now - session.last_seen >= self.idle

    def _evict_locked(self, now: datetime) -> None:
        """Drop expired sessions, then the least-recently-seen if still over the cap.

        Without a cap, every abandoned tab and every scripted login adds a row that nothing
        ever removes until its TTL runs out. The cap is generous relative to a control room's
        roster, so hitting it means something is wrong and is worth a log line.
        """
        for sid in [sid for sid, session in self._sessions.items() if self._is_expired(session, now)]:
            del self._sessions[sid]

        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda sid: self._sessions[sid].last_seen)
            evicted = self._sessions.pop(oldest)
            logger.warning(f"Session cap ({self.max_sessions}) reached; evicted '{evicted.username}'")

    def _persist_locked(self) -> None:
        """Snapshot to disk atomically. Caller holds the lock. Never raises."""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            payload = {
                "sessions": [
                    {
                        "id": sid,
                        "username": session.username,
                        "created_at": session.created_at.isoformat(),
                        "last_seen": session.last_seen.isoformat(),
                    }
                    for sid, session in self._sessions.items()
                ]
            }
            # Temp file plus `os.replace` (atomic on NTFS and POSIX): a crash mid-write must
            # not leave a truncated file that fails to parse and logs everyone out.
            temp_path = f"{self.path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp_path, self.path)
        except OSError as exc:
            logger.warning(f"Could not persist sessions: {exc}")

    def _load(self) -> None:
        """Restore sessions from the snapshot, dropping anything already expired.

        Expired rows are dropped rather than restored: reloading a session whose idle timeout
        elapsed while the process was down would resurrect exactly the session the timeout
        was meant to end.
        """
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read sessions, starting empty: {exc}")
            return

        now = datetime.now(UTC)
        restored = 0
        for entry in raw.get("sessions", []):
            try:
                session = Session(
                    username=entry["username"],
                    created_at=datetime.fromisoformat(entry["created_at"]),
                    last_seen=datetime.fromisoformat(entry["last_seen"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(f"Skipping malformed session row: {exc}")
                continue
            if self._is_expired(session, now):
                continue
            self._sessions[entry["id"]] = session
            restored += 1
        if restored:
            logger.info(f"Restored {restored} session(s)")

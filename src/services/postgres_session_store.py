"""PostgreSQL-backed session store for multi-instance deployments."""

import logging
import secrets
from datetime import UTC, datetime, timedelta

from src.services.sessions import _SESSION_ID_BYTES, Session

logger = logging.getLogger(__name__)


class PostgresSessionStore:
    def __init__(self, database_url: str, ttl_hours: float = 12.0, idle_minutes: float = 30.0, max_sessions: int = 200) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL sessions") from exc
        self._psycopg = psycopg
        self.database_url = database_url
        self.ttl = timedelta(hours=ttl_hours)
        self.idle = timedelta(minutes=idle_minutes)
        self.max_sessions = max_sessions
        self._initialize()

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    last_seen TIMESTAMPTZ NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username);
                """
            )

    def create(self, username: str) -> str:
        now = datetime.now(UTC)
        session_id = secrets.token_urlsafe(_SESSION_ID_BYTES)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE created_at <= %s OR last_seen <= %s", (now - self.ttl, now - self.idle))
            cursor.execute("SELECT session_id FROM sessions ORDER BY last_seen ASC")
            rows = cursor.fetchall()
            if len(rows) >= self.max_sessions:
                for row in rows[: len(rows) - self.max_sessions + 1]:
                    cursor.execute("DELETE FROM sessions WHERE session_id=%s", (row[0],))
            cursor.execute("INSERT INTO sessions (session_id, username, created_at, last_seen) VALUES (%s, %s, %s, %s)", (session_id, username, now, now))
        return session_id

    def resolve(self, session_id: str | None, touch: bool = True) -> Session | None:
        if not session_id:
            return None
        now = datetime.now(UTC)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT username, created_at, last_seen FROM sessions WHERE session_id=%s", (session_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            session = Session(username=row[0], created_at=row[1], last_seen=row[2])
            if now - session.created_at >= self.ttl or now - session.last_seen >= self.idle:
                cursor.execute("DELETE FROM sessions WHERE session_id=%s", (session_id,))
                return None
            if touch:
                cursor.execute("UPDATE sessions SET last_seen=%s WHERE session_id=%s", (now, session_id))
                session.last_seen = now
            return session

    def destroy(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE session_id=%s RETURNING username", (session_id,))
            row = cursor.fetchone()
        return row[0] if row else None

    def destroy_all_for(self, username: str) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE username=%s", (username,))
            return cursor.rowcount

    def active_count(self) -> int:
        now = datetime.now(UTC)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE created_at > %s AND last_seen > %s", (now - self.ttl, now - self.idle))
            return int(cursor.fetchone()[0])

"""PostgreSQL-backed operator account store."""

import logging

from src.services import db_pool
from src.services.security import DEFAULT_WORK_FACTOR, hash_password, verify_password
from src.services.user_store import User, UserRecord, _dummy_hash

logger = logging.getLogger(__name__)


class PostgresUserStore:
    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL accounts") from exc
        self._psycopg = psycopg
        self.database_url = database_url
        self._initialize()

    def _connect(self):
        return db_pool.connection(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    role TEXT NOT NULL CHECK (role IN ('guard', 'manager')),
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    @staticmethod
    def _record(row) -> UserRecord:
        return UserRecord(username=row[0], role=row[1], display_name=row[2], password_hash=row[3], active=row[4])

    def reload(self) -> None:
        return

    def authenticate(self, username: str, password: str) -> User | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT username, role, display_name, password_hash, active FROM users WHERE username=%s", (username.strip().lower(),))
            row = cursor.fetchone()
        if row is None:
            verify_password(password, _dummy_hash())
            return None
        record = self._record(row)
        if not verify_password(password, record.password_hash) or not record.active:
            return None
        return record.public()

    def get(self, username: str) -> User | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT username, role, display_name, password_hash, active FROM users WHERE username=%s AND active", (username.strip().lower(),))
            row = cursor.fetchone()
        return self._record(row).public() if row else None

    def list_users(self) -> list[User]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT username, role, display_name, password_hash, active FROM users WHERE active ORDER BY username")
            return [self._record(row).public() for row in cursor.fetchall()]

    def add_user(self, username: str, password: str, role: str, display_name: str = "", work_factor: int = DEFAULT_WORK_FACTOR) -> User:
        clean = username.strip()
        if not clean:
            raise ValueError("Username must not be empty")
        record = UserRecord(
            username=clean,
            role=role,
            display_name=display_name.strip() or clean,
            password_hash=hash_password(password, work_factor=work_factor),
        )
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (username, role, display_name, password_hash) VALUES (%s, %s, %s, %s)",
                    (clean.lower(), record.role, record.display_name, record.password_hash),
                )
        except self._psycopg.errors.UniqueViolation as exc:
            raise ValueError(f"User '{clean}' already exists") from exc
        logger.info("Created %s account '%s'", role, clean)
        return record.public()

    def import_record(self, record: UserRecord) -> None:
        """Import an existing scrypt record without rehashing its password."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO users (username, role, display_name, password_hash, active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET role=EXCLUDED.role,
                display_name=EXCLUDED.display_name, password_hash=EXCLUDED.password_hash,
                active=EXCLUDED.active""",
                (record.username.lower(), record.role, record.display_name, record.password_hash, record.active),
            )

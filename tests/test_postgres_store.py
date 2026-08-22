"""Tests against a real PostgreSQL database, not SQLite standing in for one.

Skipped by default -- these make real network round trips to whatever RUN_POSTGRES_TESTS
points at, so they are opt-in rather than part of the default `pytest -q` run:

    RUN_POSTGRES_TESTS=1 pytest tests/test_postgres_store.py -q

Reuses settings.database_url (the project's own .env) unless TEST_DATABASE_URL overrides
it, so pointing .env at a dedicated test project (never production data) is what makes this
suite safe to run. Every test cleans up exactly the rows it created, by id, rather than
truncating a table other work might be relying on.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_POSTGRES_TESTS"),
    reason="Set RUN_POSTGRES_TESTS=1 (and optionally TEST_DATABASE_URL) to run against a real Postgres",
)


def _database_url() -> str:
    from src.config import get_settings

    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    if not url.startswith(("postgresql://", "postgres://")):
        pytest.skip(f"DATABASE_URL is not a Postgres URL ({url!r}); point it at a test project first")
    return url


@pytest.fixture
def incident_store():
    from src.services.postgres_store import PostgresStore

    return PostgresStore(_database_url())


@pytest.fixture
def user_store():
    from src.services.postgres_user_store import PostgresUserStore

    return PostgresUserStore(_database_url())


@pytest.fixture
def session_store():
    from src.services.postgres_session_store import PostgresSessionStore

    return PostgresSessionStore(_database_url(), ttl_hours=12.0, idle_minutes=30.0, max_sessions=200)


def _alert_id() -> str:
    return f"test-{uuid.uuid4().hex}"


# ------------------------------------------------------------------ incidents / history
class TestPostgresStore:
    def test_insert_and_read_back_an_incident(self, incident_store):
        alert_id = _alert_id()
        try:
            incident_store.insert_incident(
                {
                    "alert_id": alert_id,
                    "camera_id": "cam_test",
                    "camera_name": "Test Camera",
                    "label": "fall_detection",
                    "confidence": 0.87,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            rows = [row for row in incident_store.incidents() if row["alert_id"] == alert_id]
            assert len(rows) == 1
            assert rows[0]["camera_id"] == "cam_test"
            assert rows[0]["label"] == "fall_detection"
            assert rows[0]["confidence"] == pytest.approx(0.87)
        finally:
            with incident_store._connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM incidents WHERE alert_id=%s", (alert_id,))

    def test_inserting_the_same_alert_id_twice_is_a_noop(self, incident_store):
        """`ON CONFLICT (alert_id) DO NOTHING` -- a retried write must not duplicate the row."""
        alert_id = _alert_id()
        row = {
            "alert_id": alert_id,
            "camera_id": "cam_test",
            "camera_name": "Test Camera",
            "label": "fall_detection",
            "confidence": 0.5,
            "created_at": datetime.now(UTC).isoformat(),
        }
        try:
            incident_store.insert_incident(row)
            incident_store.insert_incident(row)
            matches = [r for r in incident_store.incidents() if r["alert_id"] == alert_id]
            assert len(matches) == 1
        finally:
            with incident_store._connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM incidents WHERE alert_id=%s", (alert_id,))

    def test_update_incident_evidence(self, incident_store):
        alert_id = _alert_id()
        try:
            incident_store.insert_incident(
                {
                    "alert_id": alert_id,
                    "camera_id": "cam_test",
                    "camera_name": "Test Camera",
                    "label": "intrusion",
                    "confidence": 0.9,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            incident_store.update_incident_evidence(alert_id, "https://example.com/evidence.jpg")
            row = next(r for r in incident_store.incidents() if r["alert_id"] == alert_id)
            assert row["evidence_url"] == "https://example.com/evidence.jpg"
        finally:
            with incident_store._connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM incidents WHERE alert_id=%s", (alert_id,))

    def test_update_incident_review_records_the_operator_verdict(self, incident_store):
        alert_id = _alert_id()
        try:
            incident_store.insert_incident(
                {
                    "alert_id": alert_id,
                    "camera_id": "cam_test",
                    "camera_name": "Test Camera",
                    "label": "abandoned_object",
                    "confidence": 0.6,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            incident_store.update_incident_review(
                {
                    "alert_id": alert_id,
                    "operator_verdict": "real",
                    "judged_by": "gwen",
                    "reviewed_at": datetime.now(UTC).isoformat(),
                    "operator_note": "confirmed on camera",
                    "actual_event_type": "real_incident",
                }
            )
            row = next(r for r in incident_store.incidents() if r["alert_id"] == alert_id)
            assert row["operator_verdict"] == "real"
            assert row["judged_by"] == "gwen"
            assert row["actual_event_type"] == "real_incident"
        finally:
            with incident_store._connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM incidents WHERE alert_id=%s", (alert_id,))

    def test_insert_and_read_back_alert_history(self, incident_store):
        alert_id = _alert_id()
        try:
            incident_store.insert_history(
                {
                    "alert_id": alert_id,
                    "camera_id": "cam_test",
                    "label": "fall_detection",
                    "reason": "2/3 indicators",
                    "vlm_verdict": "real",
                    "vlm_confidence": 0.8,
                    "vlm_rationale": "person prone and motionless",
                    "operator_verdict": "real",
                    "judged_by": "gwen",
                    "operator_note": None,
                    "actual_event_type": "real_incident",
                    "recorded_at": datetime.now(UTC).isoformat(),
                }
            )
            rows = [r for r in incident_store.history() if r["alert_id"] == alert_id]
            assert len(rows) == 1
            assert rows[0]["vlm_verdict"] == "real"
            assert rows[0]["operator_verdict"] == "real"
        finally:
            with incident_store._connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM alert_history WHERE alert_id=%s", (alert_id,))


# ------------------------------------------------------------------ users
class TestPostgresUserStore:
    def _cleanup(self, user_store, username: str) -> None:
        with user_store._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (username.lower(),))

    def test_add_user_then_authenticate(self, user_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        try:
            user_store.add_user(username, "correct-horse-battery-staple", "guard", "Test Guard", work_factor=12)
            authenticated = user_store.authenticate(username, "correct-horse-battery-staple")
            assert authenticated is not None
            assert authenticated.username == username.lower()
            assert authenticated.role == "guard"

            assert user_store.authenticate(username, "wrong-password") is None
        finally:
            self._cleanup(user_store, username)

    def test_duplicate_username_raises(self, user_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        try:
            user_store.add_user(username, "pw", "guard", "First", work_factor=12)
            with pytest.raises(ValueError, match="already exists"):
                user_store.add_user(username, "pw2", "manager", "Second", work_factor=12)
        finally:
            self._cleanup(user_store, username)

    def test_get_and_list_users_only_return_active_accounts(self, user_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        try:
            user_store.add_user(username, "pw", "manager", "List Me", work_factor=12)
            assert user_store.get(username) is not None
            assert any(u.username == username.lower() for u in user_store.list_users())

            with user_store._connect() as conn, conn.cursor() as cur:
                cur.execute("UPDATE users SET active=FALSE WHERE username=%s", (username.lower(),))
            assert user_store.get(username) is None
            assert not any(u.username == username.lower() for u in user_store.list_users())
        finally:
            self._cleanup(user_store, username)

    def test_authenticate_unknown_username_takes_the_same_path_as_wrong_password(self, user_store):
        """No early return on a missing row -- constant-ish time, no username-enumeration tell."""
        assert user_store.authenticate(f"nobody_{uuid.uuid4().hex[:10]}", "whatever") is None


# ------------------------------------------------------------------ sessions
class TestPostgresSessionStore:
    def test_create_then_resolve(self, session_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        session_id = session_store.create(username)
        try:
            session = session_store.resolve(session_id)
            assert session is not None
            assert session.username == username
        finally:
            session_store.destroy(session_id)

    def test_resolve_missing_session_is_none(self, session_store):
        assert session_store.resolve("does-not-exist") is None
        assert session_store.resolve(None) is None

    def test_destroy_returns_the_username_and_removes_it(self, session_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        session_id = session_store.create(username)
        assert session_store.destroy(session_id) == username
        assert session_store.resolve(session_id) is None
        # Destroying again -- already gone -- returns None rather than raising.
        assert session_store.destroy(session_id) is None

    def test_destroy_all_for_a_user_removes_every_session(self, session_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        ids = [session_store.create(username) for _ in range(3)]
        try:
            removed = session_store.destroy_all_for(username)
            assert removed == 3
            assert all(session_store.resolve(sid) is None for sid in ids)
        finally:
            for sid in ids:
                session_store.destroy(sid)

    def test_expired_session_is_not_resolved(self, session_store):
        """A session past its idle window reads back as gone, not stale-but-present."""
        from src.services.postgres_session_store import PostgresSessionStore

        short_lived = PostgresSessionStore(_database_url(), ttl_hours=12.0, idle_minutes=30.0)
        username = f"test_{uuid.uuid4().hex[:10]}"
        session_id = short_lived.create(username)
        try:
            # Backdate last_seen past the idle window directly -- waiting 30 real minutes
            # is not a reasonable thing to ask a test suite to do.
            with short_lived._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET last_seen=%s WHERE session_id=%s",
                    (datetime.now(UTC) - timedelta(minutes=31), session_id),
                )
            assert short_lived.resolve(session_id) is None
        finally:
            short_lived.destroy(session_id)

    def test_active_count_reflects_current_sessions(self, session_store):
        username = f"test_{uuid.uuid4().hex[:10]}"
        before = session_store.active_count()
        session_id = session_store.create(username)
        try:
            assert session_store.active_count() == before + 1
        finally:
            session_store.destroy(session_id)
        assert session_store.active_count() == before

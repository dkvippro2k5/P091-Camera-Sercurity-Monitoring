"""PostgreSQL persistence for incidents and operator-labelled history.

The repository contract mirrors ``SQLiteStore`` so the monitoring pipeline does not know
which database is deployed. PostgreSQL is selected by a ``postgresql://`` or ``postgres://``
DATABASE_URL and requires psycopg 3 at runtime.
"""

from datetime import datetime


class PostgresStore:
    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresStore requires a PostgreSQL database URL")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - exercised in production images
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL persistence") from exc

        self._psycopg = psycopg
        self.database_url = database_url
        self._initialize()

    def _connect(self):
        return self._psycopg.connect(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS incidents (
                        alert_id TEXT PRIMARY KEY,
                        camera_id TEXT NOT NULL,
                        camera_name TEXT NOT NULL,
                        label TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        operator_verdict TEXT,
                        judged_by TEXT,
                        reviewed_at TIMESTAMPTZ,
                        operator_note TEXT,
                        actual_event_type TEXT,
                        evidence_url TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_incidents_camera_time
                        ON incidents(camera_id, created_at);
                    ALTER TABLE incidents ADD COLUMN IF NOT EXISTS evidence_url TEXT;
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id BIGSERIAL PRIMARY KEY,
                        alert_id TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        reason TEXT,
                        vlm_verdict TEXT,
                        vlm_confidence DOUBLE PRECISION,
                        vlm_rationale TEXT,
                        operator_verdict TEXT NOT NULL,
                        judged_by TEXT,
                        operator_note TEXT,
                        actual_event_type TEXT,
                        recorded_at TIMESTAMPTZ NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_history_camera_label
                        ON alert_history(camera_id, label, recorded_at);
                    """
                )

    @staticmethod
    def _timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _iso(value):
        return value.isoformat() if hasattr(value, "isoformat") else value

    def insert_incident(self, row: dict) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO incidents
                (alert_id, camera_id, camera_name, label, confidence, created_at, evidence_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (alert_id) DO NOTHING""",
                (row["alert_id"], row["camera_id"], row["camera_name"], row["label"], row["confidence"], self._timestamp(row["created_at"]), row.get("evidence_url")),
            )

    def update_incident_evidence(self, alert_id: str, evidence_url: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE incidents SET evidence_url=%s WHERE alert_id=%s", (evidence_url, alert_id))

    def update_incident_review(self, row: dict) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE incidents SET operator_verdict=%s, judged_by=%s, reviewed_at=%s,
                operator_note=%s, actual_event_type=%s WHERE alert_id=%s""",
                (row.get("operator_verdict"), row.get("judged_by"), self._timestamp(row.get("reviewed_at")), row.get("operator_note"), row.get("actual_event_type"), row["alert_id"]),
            )

    def incidents(self) -> list[dict]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT * FROM incidents ORDER BY created_at")
            columns = [item.name for item in cursor.description]
            return [{key: self._iso(value) for key, value in zip(columns, row)} for row in cursor.fetchall()]

    def insert_history(self, row: dict) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO alert_history
                (alert_id, camera_id, label, reason, vlm_verdict, vlm_confidence,
                 vlm_rationale, operator_verdict, judged_by, operator_note,
                 actual_event_type, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                tuple(row.get(key) for key in (
                    "alert_id", "camera_id", "label", "reason", "vlm_verdict",
                    "vlm_confidence", "vlm_rationale", "operator_verdict", "judged_by",
                    "operator_note", "actual_event_type",
                )) + (self._timestamp(row.get("recorded_at")),),
            )

    def history(self) -> list[dict]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT * FROM alert_history ORDER BY id")
            columns = [item.name for item in cursor.description]
            return [{key: self._iso(value) for key, value in zip(columns, row)} for row in cursor.fetchall()]

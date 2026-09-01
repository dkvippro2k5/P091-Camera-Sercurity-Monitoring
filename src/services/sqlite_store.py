"""Small SQLite persistence layer for incident and operator-review records.

The monitoring workers are synchronous, so a short-lived connection per operation is safer
than sharing a connection across worker, API, and triage threads. SQLite is the development
default; the schema is deliberately relational so it can be moved to PostgreSQL later.
"""

import os
import sqlite3
import threading
from pathlib import Path


def sqlite_path(database_url: str) -> str:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite:/// database URLs are supported by the built-in store")
    path = database_url.removeprefix("sqlite:///")
    if path.startswith("./"):
        path = path[2:]
    return os.path.abspath(path)


class SQLiteStore:
    def __init__(self, database_url: str) -> None:
        self.path = sqlite_path(database_url)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    alert_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    operator_verdict TEXT,
                    judged_by TEXT,
                    reviewed_at TEXT,
                    operator_note TEXT,
                    actual_event_type TEXT
                    ,evidence_url TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_incidents_camera_time
                    ON incidents(camera_id, created_at);
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    reason TEXT,
                    vlm_verdict TEXT,
                    vlm_confidence REAL,
                    vlm_rationale TEXT,
                    operator_verdict TEXT NOT NULL,
                    judged_by TEXT,
                    operator_note TEXT,
                    actual_event_type TEXT,
                    vlm_error TEXT,
                    recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_history_camera_label
                    ON alert_history(camera_id, label, recorded_at);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)")}
            if "evidence_url" not in columns:
                connection.execute("ALTER TABLE incidents ADD COLUMN evidence_url TEXT")
            # Added after the table shipped: CREATE TABLE IF NOT EXISTS leaves an existing
            # alert_history untouched, so the column above only reaches fresh databases.
            history_columns = {row[1] for row in connection.execute("PRAGMA table_info(alert_history)")}
            if "vlm_error" not in history_columns:
                connection.execute("ALTER TABLE alert_history ADD COLUMN vlm_error TEXT")

    def insert_incident(self, row: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO incidents
                (alert_id, camera_id, camera_name, label, confidence, created_at, evidence_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (row["alert_id"], row["camera_id"], row["camera_name"], row["label"], row["confidence"], row["created_at"], row.get("evidence_url")),
            )

    def update_incident_evidence(self, alert_id: str, evidence_url: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE incidents SET evidence_url=? WHERE alert_id=?", (evidence_url, alert_id))

    def update_incident_review(self, row: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE incidents SET operator_verdict=?, judged_by=?, reviewed_at=?,
                operator_note=?, actual_event_type=? WHERE alert_id=?""",
                (row.get("operator_verdict"), row.get("judged_by"), row.get("reviewed_at"), row.get("operator_note"), row.get("actual_event_type"), row["alert_id"]),
            )

    def incidents(self) -> list[dict]:
        with self._lock, self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM incidents ORDER BY created_at")]

    def insert_history(self, row: dict) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO alert_history
                (alert_id, camera_id, label, reason, vlm_verdict, vlm_confidence,
                 vlm_rationale, operator_verdict, judged_by, operator_note,
                 actual_event_type, vlm_error, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row.get(key) for key in (
                    "alert_id", "camera_id", "label", "reason", "vlm_verdict",
                    "vlm_confidence", "vlm_rationale", "operator_verdict", "judged_by",
                    "operator_note", "actual_event_type", "vlm_error", "recorded_at",
                )),
            )

    def history(self) -> list[dict]:
        with self._lock, self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM alert_history ORDER BY id")]

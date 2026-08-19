"""Copy incidents and operator history from SQLite into PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_postgres.py \
        --sqlite-url sqlite:///./data/app.db \
        --postgres-url postgresql://user:password@host:5432/p091
"""

import argparse
import json
from pathlib import Path

from src.services.camera_registry import FileCameraRegistry, PostgresCameraRegistry
from src.services.postgres_store import PostgresStore
from src.services.postgres_user_store import PostgresUserStore
from src.services.sqlite_store import SQLiteStore
from src.services.user_store import UserRecord


def migrate(sqlite_url: str, postgres_url: str, users_path: str, cameras_path: str) -> tuple[int, int, int, int]:
    source = SQLiteStore(sqlite_url)
    target = PostgresStore(postgres_url)

    incidents = source.incidents()
    for row in incidents:
        target.insert_incident(row)
        if row.get("operator_verdict"):
            target.update_incident_review(
                {
                    "alert_id": row["alert_id"],
                    "operator_verdict": row.get("operator_verdict"),
                    "judged_by": row.get("judged_by"),
                    "reviewed_at": row.get("reviewed_at"),
                    "operator_note": row.get("operator_note"),
                    "actual_event_type": row.get("actual_event_type"),
                }
            )

    history = source.history()
    for row in history:
        target.insert_history(row)

    users = 0
    if Path(users_path).exists():
        target_users = PostgresUserStore(postgres_url)
        with open(users_path, encoding="utf-8") as handle:
            for raw in json.load(handle).get("users", []):
                target_users.import_record(UserRecord(**raw))
                users += 1

    cameras = 0
    if Path(cameras_path).exists():
        saved = FileCameraRegistry(cameras_path).load()
        PostgresCameraRegistry(postgres_url).save(saved)
        cameras = len(saved)

    return len(incidents), len(history), users, cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate P-091 event data from SQLite to PostgreSQL")
    parser.add_argument("--sqlite-url", default="sqlite:///./data/app.db")
    parser.add_argument("--postgres-url", required=True)
    parser.add_argument("--users-path", default="data/users.json")
    parser.add_argument("--cameras-path", default="data/cameras.json")
    args = parser.parse_args()

    incident_count, history_count, users, cameras = migrate(args.sqlite_url, args.postgres_url, args.users_path, args.cameras_path)
    print(f"Migrated {incident_count} incidents, {history_count} history rows, {users} users, and {cameras} cameras")


if __name__ == "__main__":
    main()

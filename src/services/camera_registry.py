"""Durable camera configuration repositories for local and production deployments."""

import json
import logging
import os

from src.config import get_settings
from src.services import db_pool

logger = logging.getLogger(__name__)


class FileCameraRegistry:
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as handle:
                return json.load(handle).get("cameras", [])
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Ignoring unreadable camera registry: %s", exc)
            return []

    def save(self, cameras: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary = f"{self.path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"cameras": cameras}, handle, indent=2)
        os.replace(temporary, self.path)


class PostgresCameraRegistry:
    def __init__(self, database_url: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self.database_url = database_url
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS camera_configs (
                    camera_id TEXT PRIMARY KEY,
                    config JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def _connect(self):
        return db_pool.connection(self.database_url)

    def load(self) -> list[dict]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT config FROM camera_configs ORDER BY camera_id")
            return [row[0] for row in cursor.fetchall()]

    def save(self, cameras: list[dict]) -> None:
        from psycopg.types.json import Jsonb

        ids = [camera["id"] for camera in cameras]
        with self._connect() as connection, connection.cursor() as cursor:
            for camera in cameras:
                cursor.execute(
                    """INSERT INTO camera_configs (camera_id, config, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (camera_id) DO UPDATE SET config=EXCLUDED.config, updated_at=NOW()""",
                    (camera["id"], Jsonb(camera)),
                )
            if ids:
                cursor.execute("DELETE FROM camera_configs WHERE NOT (camera_id = ANY(%s))", (ids,))
            else:
                cursor.execute("DELETE FROM camera_configs")


def create_camera_registry(path: str):
    """Select the camera-config store from DATABASE_URL, falling back to a local JSON file.

    Selected by DATABASE_URL, not APP_ENV -- the same fix already applied to `event_store.py`
    and `embedding_store.py`. Gating on which environment string happens to be set, rather
    than on what the configured database actually is, meant a fully-configured Postgres URL
    outside "production" was silently ignored and the operator's camera setup only ever
    landed in `data/cameras.json` -- while every other durable table went to Postgres.
    """
    settings = get_settings()
    if settings.database_url.startswith(("postgresql://", "postgres://")):
        return PostgresCameraRegistry(settings.database_url)
    return FileCameraRegistry(path)

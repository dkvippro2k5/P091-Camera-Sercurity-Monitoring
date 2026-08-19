"""Select the durable incident store from DATABASE_URL."""

from src.config import get_settings
from src.services.sqlite_store import SQLiteStore


def create_event_store(database_url: str | None = None):
    settings = get_settings()
    url = database_url or settings.database_url
    if settings.app_env.lower() == "production" and url.startswith(("postgresql://", "postgres://")):
        from src.services.postgres_store import PostgresStore

        return PostgresStore(url)
    return SQLiteStore(url)

"""Optional semantic incident storage backed by PostgreSQL and pgvector.

Embedding generation is intentionally outside this module and outside the live camera
path. Callers can enqueue text/image embedding work to a separate job system, then use
this repository for durable storage and similarity search.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.config import get_settings
from src.services import db_pool

logger = logging.getLogger(__name__)


class PostgresEmbeddingStore:
    """Store incident embeddings in PostgreSQL with the pgvector extension."""

    def __init__(self, database_url: str, dimensions: int = 1536) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresEmbeddingStore requires a PostgreSQL database URL")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL embeddings") from exc
        self._psycopg = psycopg
        self.database_url = database_url
        self.dimensions = dimensions
        self._initialize()

    def _connect(self):
        return db_pool.connection(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS incident_embeddings (
                    alert_id TEXT PRIMARY KEY REFERENCES incidents(alert_id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )"""
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_incident_embeddings_vector "
                "ON incident_embeddings USING hnsw (embedding vector_cosine_ops)"
            )

    def upsert(self, alert_id: str, content: str, embedding: list[float], model: str) -> None:
        if len(embedding) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(embedding)}")
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO incident_embeddings(alert_id, content, embedding, model)
                   VALUES (%s, %s, %s::vector, %s)
                   ON CONFLICT (alert_id) DO UPDATE SET content=EXCLUDED.content,
                     embedding=EXCLUDED.embedding, model=EXCLUDED.model""",
                (alert_id, content, vector, model),
            )

    def search(self, embedding: list[float], limit: int = 10, exclude_alert_id: str | None = None) -> list[dict]:
        """Nearest confirmed incidents by cosine similarity, joined with the fields the UI needs.

        Joins `incidents` rather than returning just `content`+`similarity`: the caller (the
        guard looking at a live alert, or the VLM prompt) needs to know *what happened* with
        each neighbor -- which camera, what the guard decided -- not just that it was similar.
        `exclude_alert_id` matters once this alert's own embedding is later stored: without
        it, an alert judged and then queried immediately after would find itself as its own
        nearest (identical) neighbor.
        """
        if len(embedding) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(embedding)}")
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT e.alert_id, e.content, i.camera_id, i.camera_name, i.label,
                          i.operator_verdict, i.evidence_url, e.created_at,
                          1 - (e.embedding <=> %s::vector) AS similarity
                   FROM incident_embeddings e
                   JOIN incidents i ON i.alert_id = e.alert_id
                   WHERE %s::text IS NULL OR e.alert_id != %s
                   ORDER BY e.embedding <=> %s::vector
                   LIMIT %s""",
                (vector, exclude_alert_id, exclude_alert_id, vector, max(1, min(limit, 100))),
            )
            return [
                {
                    "alert_id": row[0],
                    "content": row[1],
                    "camera_id": row[2],
                    "camera_name": row[3],
                    "label": row[4],
                    "operator_verdict": row[5],
                    "evidence_url": row[6],
                    "created_at": row[7].isoformat() if isinstance(row[7], datetime) else row[7],
                    "similarity": float(row[8]),
                }
                for row in cursor.fetchall()
            ]


def create_embedding_store() -> PostgresEmbeddingStore | None:
    """Build the repository only when embeddings are enabled and the DB is Postgres.

    Selected by DATABASE_URL, not APP_ENV -- the same fix applied to `event_store.py` and
    `main.py`'s `use_postgres` for the same reason: gating persistence on which environment
    string happens to be set, rather than on what the configured database actually is, means
    a fully-configured Postgres URL in a non-"production" environment gets silently ignored.
    pgvector is Postgres-only, so unlike the other stores there is no local-file fallback to
    degrade to -- if the URL is not Postgres, this feature is simply off.
    """
    settings = get_settings()
    if not settings.embeddings_enabled:
        return None
    if not settings.database_url.startswith(("postgresql://", "postgres://")):
        logger.warning("EMBEDDINGS_ENABLED is true but DATABASE_URL is not Postgres; similar-incident lookup is disabled.")
        return None
    return PostgresEmbeddingStore(settings.database_url, settings.embedding_dimensions)

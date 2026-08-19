"""Optional semantic incident storage backed by PostgreSQL and pgvector.

Embedding generation is intentionally outside this module and outside the live camera
path. Callers can enqueue text/image embedding work to a separate job system, then use
this repository for durable storage and similarity search.
"""

from __future__ import annotations

from datetime import datetime

from src.config import get_settings


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
        return self._psycopg.connect(self.database_url)

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

    def search(self, embedding: list[float], limit: int = 10) -> list[dict]:
        if len(embedding) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(embedding)}")
        vector = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT alert_id, content, model, created_at,
                          1 - (embedding <=> %s::vector) AS similarity
                   FROM incident_embeddings ORDER BY embedding <=> %s::vector LIMIT %s""",
                (vector, vector, max(1, min(limit, 100))),
            )
            return [
                {"alert_id": row[0], "content": row[1], "model": row[2],
                 "created_at": row[3].isoformat() if isinstance(row[3], datetime) else row[3],
                 "similarity": float(row[4])}
                for row in cursor.fetchall()
            ]


def create_embedding_store() -> PostgresEmbeddingStore | None:
    """Build the production repository only when embeddings are explicitly enabled."""
    settings = get_settings()
    if not settings.embeddings_enabled:
        return None
    if settings.app_env.lower() != "production":
        return None
    if not settings.database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("Embeddings in production require a PostgreSQL DATABASE_URL")
    return PostgresEmbeddingStore(settings.database_url, settings.embedding_dimensions)

"""One shared connection pool per database URL, for every Postgres-backed store.

Opening a fresh `psycopg.connect()` per store method was fine while `DATABASE_URL` pointed
at a local SQLite file and the Postgres path was dead code. Against a real hosted Postgres
it is not: a measured **1.1 s** per connect+`SELECT 1` round trip to Supabase ap-southeast-1,
almost all of it TCP + TLS + auth handshake rather than the query.

That cost is what made the dashboard look frozen. Every store call — recording an alert on a
camera's processor thread, resolving a session on each wake of the SSE loop, the three writes
behind one operator verdict — spent a second or more holding a worker thread. `asyncio.to_thread`
keeps the event loop free but draws from a default pool of only `min(32, cpu_count + 4)` threads,
so an alert storm across six cameras saturated *that* instead, and every subsequent login,
verdict, and `/auth/me` queued behind a handshake nobody needed to repeat.

A pool pays the handshake once and hands the same connection back. Sizing and timeouts:

- `max_size` is deliberately small. Supabase's direct (non-pooler) endpoint allows only tens
  of server-side connections, shared with psql sessions, migrations, and the dashboard; a
  pool that tries to own them all turns a slow app into a broken one.
- `timeout` bounds the wait for a *free pooled* connection, so a saturated pool raises rather
  than parking a worker thread forever — which is the failure mode being fixed here.
- `connect_timeout` bounds the underlying TCP connect, so an unreachable host fails in seconds
  instead of hanging on the OS default.
- `check=ConnectionPool.check_connection` because pooled connections outlive Supabase's own
  idle reaping; without it the first query after an idle spell fails on a dead socket.
"""

from __future__ import annotations

import atexit
import logging
import threading

logger = logging.getLogger(__name__)

_POOL_MAX_SIZE = 8
_POOL_WAIT_SECONDS = 10.0
_CONNECT_TIMEOUT_SECONDS = 10

_pools: dict[str, object] = {}
_lock = threading.Lock()


def get_pool(database_url: str):
    """Return the process-wide pool for `database_url`, creating it on first use.

    Keyed by URL so the incident, user, and session stores — constructed separately but
    always pointed at the same database — share one set of connections rather than three.
    """
    with _lock:
        pool = _pools.get(database_url)
        if pool is not None:
            return pool

        try:
            from psycopg_pool import ConnectionPool
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency is in requirements
            raise RuntimeError("Install psycopg[binary,pool] to use PostgreSQL persistence") from exc

        pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=_POOL_MAX_SIZE,
            timeout=_POOL_WAIT_SECONDS,
            max_idle=300.0,
            kwargs={"connect_timeout": _CONNECT_TIMEOUT_SECONDS},
            check=ConnectionPool.check_connection,
            # Not opened in the constructor: that would make importing a store enough to
            # reach out to the network, which breaks test collection on a machine with no
            # database. `open()` below is explicit and happens only once we are really used.
            open=False,
            name="ai20k",
        )
        pool.open()
        _pools[database_url] = pool
        logger.info("Opened Postgres connection pool (max_size=%d)", _POOL_MAX_SIZE)
        return pool


def close_all() -> None:
    """Close every pool.

    Registered with `atexit`, and called from the app's lifespan on shutdown so that a
    `--reload` cycle hands its server-side connections back instead of leaving them for
    Postgres to time out — on Supabase's direct endpoint that budget is small enough for
    a few reloads to exhaust it.
    """
    with _lock:
        for pool in _pools.values():
            try:
                pool.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.exception("Failed to close a Postgres connection pool")
        _pools.clear()


atexit.register(close_all)

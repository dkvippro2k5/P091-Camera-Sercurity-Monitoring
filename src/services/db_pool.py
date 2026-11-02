"""Shared PostgreSQL connection pools, keyed by connection string.

Every Postgres repository in this package used to call `psycopg.connect()` per method, so
each query paid a fresh TCP handshake, TLS negotiation and authentication round trip. That
is affordable against a Postgres on the same docker network and expensive against a managed
one: `PostgresSessionStore.resolve` already documents ~0.9s per authenticated request
measured against Supabase ap-southeast-1, and connection setup is a large part of it. Worse,
connect-per-call scales the *number of open connections* with concurrency — several camera
workers plus the triage and embedding pools can exhaust a managed instance's connection
allowance, which surfaces as intermittent failures rather than as slowness.

The pools live here rather than on each store because five repositories share one database.
Giving each its own pool would multiply the idle connections held against that allowance for
no benefit; keying on the connection string means they all draw from the same set.

`connection()` is a drop-in for `psycopg.connect()` at the call sites, which all use it as
`with self._connect() as conn, conn.cursor() as cur:`. `pool.connection()` keeps the same
contract — commit on clean exit, rollback on exception — and differs only in returning the
connection to the pool instead of closing it.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Held against the database server's connection allowance, so deliberately modest. A managed
# Postgres typically permits far fewer direct connections than a self-hosted one, and this
# process only ever has a handful of DB-touching threads (request handlers, the triage pool,
# the embedding pool). Raise only with a measured reason.
_MAX_POOL_SIZE = 10

# One connection is kept warm so the first request after an idle period does not pay for a
# cold connect; the rest are opened on demand.
_MIN_POOL_SIZE = 1

# How long a caller waits for a free connection before failing. Bounded so a saturated pool
# surfaces as a prompt error instead of a request that appears to hang.
_POOL_TIMEOUT_SEC = 10.0

# Idle connections are closed after this long. Managed databases and the middleboxes in front
# of them drop connections that sit unused, and a connection reaped on our side is one that
# cannot be handed out already dead. Kept below the usual server-side idle timeouts.
_MAX_IDLE_SEC = 120.0

# Hosts that front Postgres with a transaction-mode pooler (PgBouncer and friends). Behind
# one, a client connection is not a server session: consecutive statements can land on
# different backends, so anything with server-side session state breaks.
_TRANSACTION_POOLER_MARKERS = ("pooler.supabase.com", "pgbouncer")


def _is_transaction_pooler(conninfo: str) -> bool:
    """Whether this URL points at a transaction-mode pooler rather than at Postgres itself.

    Matters because psycopg3 prepares a statement automatically once it has been seen five
    times on a connection, naming it `_pg3_N` with a per-connection counter. Without a pool
    every query got a fresh connection, so that counter always restarted and the feature was
    invisible. Pooling makes connections long-lived, and behind a transaction pooler the name
    the client believes is free may already exist on the backend it lands on -- surfacing as
    `DuplicatePreparedStatement: prepared statement "_pg3_2" already exists` on a route that
    had worked for hours.
    """
    return any(marker in conninfo for marker in _TRANSACTION_POOLER_MARKERS)

_pools: dict[str, object] = {}
_pools_lock = threading.Lock()
# Set once, when psycopg_pool turns out to be unavailable, so the warning is logged once
# rather than on every query.
_pool_unavailable = False


def _get_pool(conninfo: str):
    """Return the shared pool for this connection string, creating it on first use.

    Returns None when pooling is unavailable, which makes `connection()` fall back to an
    unpooled connect. An optional dependency missing must degrade performance, never take
    the application down — the same reasoning the stores apply to psycopg itself.
    """
    global _pool_unavailable

    if _pool_unavailable:
        return None

    pool = _pools.get(conninfo)
    if pool is not None:
        return pool

    with _pools_lock:
        # Re-check inside the lock: two threads can miss above and both arrive here.
        pool = _pools.get(conninfo)
        if pool is not None:
            return pool

        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            _pool_unavailable = True
            logger.warning(
                "psycopg_pool is not installed; falling back to one connection per query. "
                "Install psycopg-pool to enable connection pooling."
            )
            return None

        behind_pooler = _is_transaction_pooler(conninfo)

        def _configure(conn) -> None:
            """Disable automatic prepared statements behind a transaction pooler.

            `prepare_threshold=None` stops psycopg3 from ever preparing, which is what makes
            the pool safe behind PgBouncer in transaction mode. The cost is losing the plan
            reuse on repeated statements; the alternative is intermittent 500s, and every
            query here is a small indexed lookup where planning is not the expensive part.
            """
            conn.prepare_threshold = None

        # `check` costs one cheap round trip per checkout and buys transparent recovery from
        # a connection the server closed while it sat in the pool. Without it that stale
        # connection reaches the caller as an OperationalError — on this codebase's paths
        # that means a spurious 500 on an authenticated request, which is worse than the
        # milliseconds the check costs.
        #
        # open=False plus an explicit open() avoids the constructor-opens-implicitly
        # behaviour that psycopg_pool deprecates.
        pool = ConnectionPool(
            conninfo,
            min_size=_MIN_POOL_SIZE,
            max_size=_MAX_POOL_SIZE,
            timeout=_POOL_TIMEOUT_SEC,
            max_idle=_MAX_IDLE_SEC,
            check=ConnectionPool.check_connection,
            configure=_configure if behind_pooler else None,
            name="p091",
            open=False,
        )
        pool.open()
        _pools[conninfo] = pool
        logger.info(
            f"Opened Postgres connection pool (max_size={_MAX_POOL_SIZE}, "
            f"prepared_statements={'off (transaction pooler)' if behind_pooler else 'on'})"
        )
        return pool


@contextmanager
def connection(conninfo: str) -> Iterator:
    """A pooled connection, used exactly like `psycopg.connect()` in a `with` block."""
    pool = _get_pool(conninfo)
    if pool is None:
        import psycopg

        with psycopg.connect(conninfo) as conn:
            yield conn
        return

    with pool.connection() as conn:
        yield conn


def close_all() -> None:
    """Close every pool. Idempotent, so the atexit hook and an explicit shutdown may both run."""
    with _pools_lock:
        pools = list(_pools.items())
        _pools.clear()

    for conninfo, pool in pools:
        try:
            pool.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.warning(f"Error closing Postgres pool: {type(exc).__name__}: {exc}")
    if pools:
        logger.info(f"Closed {len(pools)} Postgres connection pool(s)")


# Covers the paths that never run the FastAPI lifespan shutdown — CLI scripts, the benchmark
# runner, and tests that construct a store directly.
atexit.register(close_all)

-- PostgreSQL schema actually created by the running application.
--
-- This file describes reality, not a plan: every statement below is copied verbatim from
-- the `CREATE TABLE` / `CREATE INDEX` calls each store already runs against DATABASE_URL
-- on startup (`_initialize()` in each file, guarded by `CREATE TABLE IF NOT EXISTS`, so
-- running the app is itself idempotent schema migration -- this file is not applied by a
-- migration tool and nothing reads it at runtime). It exists so a reviewer, or anyone
-- provisioning a fresh database by hand, can see the whole schema in one place instead of
-- across three files.
--
-- Source of truth, kept in sync with this file by hand:
--   src/services/postgres_store.py         (incidents, alert_history)
--   src/services/postgres_user_store.py    (users)
--   src/services/postgres_session_store.py (sessions)
--
-- An earlier version of this file (commit 8313750, 2026-08-06) designed a normalized
-- schema -- separate `events`/`incidents` tables, UUID keys, an `email` column on `users`,
-- pgvector embeddings -- before any of it was implemented. The schema below is what got
-- built instead: flatter, and shaped by what the FastAPI layer actually needs to read and
-- write, one row per alert rather than a join across tables. Superseded, not corrected --
-- if a later redesign wants the normalized shape, that is a real migration with a plan for
-- the rows already in `incidents`, not a doc edit.
--
-- Not covered here: SQLite is the other implementation of the same repository contract
-- (`SqliteStore`, `UserStore`, `SessionStore` -- see src/services/{sqlite_store,user_store,
-- sessions}.py), selected automatically when DATABASE_URL is not a postgresql:// URL. It
-- keeps the same column shapes in its own local files; this document only covers the
-- Postgres path, which is what a real deployment uses.


-- =========================================================================
-- users -- operator accounts (guards and managers)
-- Created by: PostgresUserStore._initialize() (src/services/postgres_user_store.py)
-- =========================================================================
CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,                          -- lowercased before storage; login is case-insensitive
    role            TEXT NOT NULL CHECK (role IN ('guard', 'manager')),
    display_name    TEXT NOT NULL,
    password_hash   TEXT NOT NULL,                              -- scrypt, formatted by src/services/security.py
    active          BOOLEAN NOT NULL DEFAULT TRUE,               -- FALSE = disabled; authenticate()/get() both filter on this
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =========================================================================
-- sessions -- server-side session store behind the auth cookie
-- Created by: PostgresSessionStore._initialize() (src/services/postgres_session_store.py)
-- =========================================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,                           -- secrets.token_urlsafe, carried in the HttpOnly cookie
    username        TEXT NOT NULL,                               -- no FK to users: a deleted account should not orphan-cascade a session lookup; deps.py handles a dangling username by treating it as signed-out
    created_at      TIMESTAMPTZ NOT NULL,                        -- absolute TTL boundary (session_ttl_hours)
    last_seen       TIMESTAMPTZ NOT NULL                         -- idle-timeout boundary (session_idle_minutes); NOT touched by the long-lived-stream liveness poll, only by an actual request
);

CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username);


-- =========================================================================
-- incidents -- one row per alert, updated in place as it gets reviewed
-- Created by: PostgresStore._initialize() (src/services/postgres_store.py)
--
-- Not split into a separate "detection event" + "incident" table: the rule engine already
-- decides what counts as alert-worthy before this table is touched, so there is exactly one
-- row per Alert, inserted once and updated twice more (evidence URL, then operator verdict)
-- as the same alert's lifecycle proceeds -- never a join.
-- =========================================================================
CREATE TABLE IF NOT EXISTS incidents (
    alert_id            TEXT PRIMARY KEY,                        -- uuid4 hex, matches Alert.id
    camera_id           TEXT NOT NULL,
    camera_name         TEXT NOT NULL,                           -- denormalized at insert time so a renamed/deleted camera does not rewrite history
    label               TEXT NOT NULL,                           -- 'fall_detection' | 'intrusion' | 'abandoned_object' (RuleEngine event types; not DB-enforced)
    confidence          DOUBLE PRECISION NOT NULL,                -- rule engine's own confidence, not the VLM's
    created_at          TIMESTAMPTZ NOT NULL,
    operator_verdict    TEXT,                                    -- 'real' | 'false', NULL until a guard judges it (OperatorVerdict in camera_schemas.py)
    judged_by           TEXT,                                    -- username of the guard who judged it; NULL on rows judged before accounts existed
    reviewed_at         TIMESTAMPTZ,
    operator_note       TEXT,                                    -- free text, guard-entered, max 2000 chars enforced at the API layer only
    actual_event_type   TEXT,                                    -- guard's own label for what really happened, e.g. 'real_incident' | 'false_alarm' | 'normal_activity' | 'maintenance' | 'other' -- free text at the DB layer, options fixed only in the frontend dropdown
    evidence_url        TEXT                                     -- set asynchronously once the evidence pool finishes encoding the incident frame; NULL briefly after insert
);

CREATE INDEX IF NOT EXISTS idx_incidents_camera_time ON incidents(camera_id, created_at);

-- Present in the running code as a defensive ALTER even though CREATE TABLE above already
-- declares the column: a database created before evidence_url existed needs this to catch
-- up. Harmless no-op on a fresh database; kept here so this file matches _initialize() line
-- for line rather than silently dropping a statement that still runs in production.
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS evidence_url TEXT;


-- =========================================================================
-- alert_history -- append-only, one row per operator-judged alert
-- Created by: PostgresStore._initialize() (src/services/postgres_store.py)
--
-- Distinct from `incidents` on purpose: `incidents` is mutable (rows get updated as an
-- alert is reviewed), this is the immutable training/audit record written once
-- (record_operator_verdict) at the same time as the incidents update. It is what
-- eval_report.py and the manager statistics dashboard read from -- an update-in-place
-- table cannot answer "what did the VLM say vs. what did the guard decide" once the
-- VLM's field has since been overwritten by a later triage run on the same alert.
-- =========================================================================
CREATE TABLE IF NOT EXISTS alert_history (
    id                  BIGSERIAL PRIMARY KEY,
    alert_id            TEXT NOT NULL,                           -- not a FK/unique: same alert_id could theoretically be judged twice if record_operator_verdict is called again; the log keeps every call, incidents keeps only the latest
    camera_id           TEXT NOT NULL,
    label               TEXT NOT NULL,
    reason              TEXT,                                    -- RuleEngine's human-readable indicator breakdown (see rule_engine.py's _fall_indicators docstring)
    vlm_verdict         TEXT,                                    -- 'real' | 'false_positive' | 'uncertain', NULL if triage was off or skipped (TriageVerdictLabel)
    vlm_confidence      DOUBLE PRECISION,
    vlm_rationale       TEXT,
    operator_verdict    TEXT NOT NULL,                           -- 'real' | 'false' -- the one field that is never NULL, since a history row only exists once a guard has judged
    judged_by           TEXT,
    operator_note       TEXT,
    actual_event_type   TEXT,
    recorded_at         TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_camera_label ON alert_history(camera_id, label, recorded_at);


-- =========================================================================
-- ============================ PLANNED -- NOT YET WIRED IN ===============
-- =========================================================================
-- Everything below this line does not run today. `PostgresEmbeddingStore`
-- (src/services/embedding_store.py) already implements this table, but nothing in src/
-- constructs or calls it -- confirmed 2026-08-24 (`grep -rn PostgresEmbeddingStore src/`
-- outside the file itself: no hits). Recorded here as the agreed design for the
-- "similar past incidents" feature (HITL-driven false-alarm reduction), so the schema and
-- the decision behind it are reviewable together before implementation starts.
--
-- What it is for: today, `AlertHistory.recent_examples()` feeds the VLM a handful of past
-- judged incidents, but only exact matches on (camera_id, label) -- a fall on camera A
-- teaches the model nothing about a geometrically identical false alarm recorded on
-- camera B. This table lets that comparison run on meaning instead of exact keys: each
-- judged incident's rule-engine `reason` text is embedded, and a new alert can be compared
-- against every past judged incident by vector distance, not just ones sharing its camera
-- and label.
--
-- Decisions made 2026-08-24 (see chat, not re-litigated by a later reader without cause):
--   - Scope is system-wide: no camera_id/label filter on the similarity query. A "person
--     bent down and stood back up quickly" case should surface as similar to a fall alert
--     even if it happened on a different camera watching a different label.
--   - The comparison library starts EMPTY and grows only from judged incidents recorded
--     after this feature ships -- no backfill of alert_history's existing rows. Simpler,
--     and avoids seeding the library with cases whose `reason` text was produced by
--     whatever rule-engine version was live when they were recorded, which may not match
--     the current one's phrasing closely enough to embed comparably.
--   - `content` is the RuleEngine `reason` string alone (already generated for every
--     alert, no extra cost to produce). Visual similarity via image embeddings (CLIP-style)
--     was explicitly deferred, not rejected -- a real v2, not this table's job.
--   - Embedding provider: the project's existing Qwen/DashScope-intl key, model
--     `text-embedding-v3` -- verified live 2026-08-24 against the same OPENAI_BASE_URL
--     already used for VLM triage (1024-dimensional output). No new credential needed.
--     `dimensions` below must match whatever model is actually configured; 1024 is
--     text-embedding-v3's own output size, not a chosen default -- if the provider or
--     model changes, this column's width has to be migrated, not just the app config.
--
-- Write path (planned): PostgresStore.record_operator_verdict() embeds `reason` and
-- upserts here in the same transaction as the alert_history insert, so the comparison
-- library only ever contains guard-confirmed cases -- never a raw, unjudged alert.
--
-- Read path (planned): a new `GET /alerts/{id}/similar` endpoint embeds the new alert's
-- `reason` on the fly (not stored -- it is not yet judged) and queries this table for the
-- nearest confirmed neighbors, for two consumers: the VLM triage prompt (replacing or
-- extending recent_examples' same-camera-and-label few-shot list with these instead) and
-- a "Similar past events" panel in AlertModal.jsx for the guard to see directly.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS incident_embeddings (
    alert_id    TEXT PRIMARY KEY REFERENCES incidents(alert_id) ON DELETE CASCADE,  -- one embedding per judged incident; deleting the incident drops its embedding too
    content     TEXT NOT NULL,          -- the RuleEngine `reason` string that was embedded, kept alongside the vector so a result can be shown/debugged without re-embedding
    embedding   vector(1024) NOT NULL,  -- text-embedding-v3 output width -- see note above before changing
    model       TEXT NOT NULL,          -- embedding model name, stamped per row so a future model change does not silently mix incomparable vectors under one index
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW: approximate nearest-neighbor, sublinear query time as the table grows, at the cost
-- of the index itself being approximate (may occasionally miss the true nearest neighbor).
-- Acceptable here -- "5 similar past cases" does not need to be the mathematically exact 5
-- out of however many thousand incidents accumulate. `vector_cosine_ops` because embedding
-- similarity is conventionally compared by angle (cosine), not raw Euclidean distance.
CREATE INDEX IF NOT EXISTS idx_incident_embeddings_vector
    ON incident_embeddings USING hnsw (embedding vector_cosine_ops);

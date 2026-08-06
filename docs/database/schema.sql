-- Schema cho Hệ Thống AI Agent Giám Sát Ngữ Cảnh & Phản Ứng Sự Cố An Ninh Khu Đô Thị
-- Chạy tự động khi container db khởi tạo lần đầu (docker-entrypoint-initdb.d)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =========================
-- 1. USERS (bảo vệ / quản lý)
-- =========================
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(150) UNIQUE NOT NULL,
    role        VARCHAR(20) NOT NULL CHECK (role IN ('guard', 'manager')),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- =========================
-- 2. CAMERAS
-- =========================
CREATE TABLE cameras (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name              VARCHAR(100) NOT NULL,
    location          VARCHAR(200),
    rtsp_url          TEXT NOT NULL,
    restricted_zone   JSONB,
    restricted_hours  JSONB,
    is_active         BOOLEAN DEFAULT true,
    created_at        TIMESTAMPTZ DEFAULT now()
);

-- =========================
-- 3. ZONE_RULES (ngưỡng theo từng khu vực / nhãn)
-- =========================
CREATE TABLE zone_rules (
    id                           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id                    UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    event_type                   VARCHAR(30) NOT NULL CHECK (event_type IN ('intrusion','fall','abandoned_object')),
    abandon_timeout_seconds      INT,
    owner_proximity_radius       NUMERIC(5,2),
    fall_still_duration_seconds  INT,
    intrusion_confirm_frames     INT,
    created_at                   TIMESTAMPTZ DEFAULT now()
);

-- =========================
-- 4. EVENTS (dữ liệu thô từ Vision Engineer)
-- =========================
CREATE TABLE events (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id        UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    event_type       VARCHAR(30) NOT NULL CHECK (event_type IN ('intrusion','fall','abandoned_object')),
    track_id         VARCHAR(50),
    confidence       NUMERIC(4,3),
    frame_timestamp  TIMESTAMPTZ NOT NULL,
    frame_url        TEXT,
    detection_data   JSONB NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_events_camera_time ON events (camera_id, frame_timestamp DESC);
CREATE INDEX idx_events_type ON events (event_type);
CREATE INDEX idx_events_track_id ON events (track_id);

-- =========================
-- 5. INCIDENTS (sau khi AI Agent đánh giá ngữ cảnh)
-- =========================
CREATE TABLE incidents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id        UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    severity        VARCHAR(10) NOT NULL CHECK (severity IN ('low','medium','high')),
    llm_reasoning   TEXT,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','confirmed','dismissed')),
    confirmed_by    UUID REFERENCES users(id),
    confirmed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_incidents_status ON incidents (status);
CREATE INDEX idx_incidents_event ON incidents (event_id);

-- =========================
-- 6. INCIDENT_CONTEXT_EMBEDDINGS (pgvector)
-- =========================
CREATE TABLE incident_context_embeddings (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id   UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    embedding     vector(1536),
    metadata      JSONB,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_embedding_cosine ON incident_context_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

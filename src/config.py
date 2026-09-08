from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "AI20K Agent"
    app_env: str = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: str = "INFO"
    # Vite's dev server defaults to 5173; without it the browser blocks every fetch to
    # this API with an opaque CORS error that looks like the backend is down.
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    allowed_hosts: str = "localhost,127.0.0.1,test,testserver"
    # Whether startup re-adds the cameras saved in the registry. On by default, because a
    # deployment that restarts must come back watching what it was watching. Turn it off for
    # local work, where the saved set is whatever the last experiment left behind and having
    # it start itself means every `--reload` re-opens six video files and floods the alert
    # queue before anyone has asked for a camera at all.
    restore_cameras_on_start: bool = True

    # LLM
    openai_api_key: str = ""
    # Empty keeps the official OpenAI endpoint. Set to point the same OpenAI-compatible
    # client at a proxy/alternate provider (e.g. an internal gateway) without touching code.
    openai_base_url: str = ""
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # VLM triage (PRD §1.2: LLM context reasoning to filter false alarms)
    # Off by default, so a fresh clone and CI both behave exactly as they did before this
    # layer existed. The local provider does not require an API key.
    vlm_enabled: bool = False
    vlm_provider: Literal["openai", "local"] = "openai"
    vlm_model: str = "gpt-4o-mini"
    vlm_local_model: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm_local_device: str = "auto"
    vlm_local_max_image_size: int = Field(default=512, ge=224, le=1024)
    vlm_local_max_new_tokens: int = Field(default=64, ge=16, le=256)
    # Pre-roll must reach back past RuleEngine.fall_dwell_sec (3.0) plus the 0.5s velocity
    # window. Anything shorter and the "before" frames show a person already on the ground,
    # so the model sees no fall and correctly calls a real fall a false positive.
    vlm_pre_roll_sec: float = Field(default=5.0, gt=0.0)
    vlm_post_roll_sec: float = Field(default=2.0, ge=0.0)
    # Every frame is a billed image and accuracy degrades with long image sequences.
    vlm_max_frames: int = Field(default=6, ge=2, le=16)
    vlm_timeout_sec: float = Field(default=20.0, gt=0.0)
    vlm_history_examples: int = Field(default=6, ge=0, le=32)

    # Monitoring
    # Per (camera, label) floor between alerts. One flickering track would otherwise
    # open a modal on every frame it flickers.
    alert_cooldown_sec: float = Field(default=30.0, ge=0.0)
    # Abandoned-object rule. 2s is a demo/illustration default so a short clip shows the
    # alert fire; PETS2006/AVSS-style production deployments use 30-60s. Not a RuleEngine
    # default because that module is pinned by its test suite — this is the one construction
    # site that reads Settings, and stream_manager already copies these onto every worker.
    abandon_sec: float = Field(default=2.0, gt=0.0)
    owner_radius_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    stationary_tolerance_ratio: float = Field(default=0.02, gt=0.0, le=1.0)
    camera_target_fps: float = Field(default=5.0, gt=0.0, le=60.0)
    # Zero selects a host-aware default (1 concurrent inference on GPU, up to 2 on CPU).
    camera_max_concurrent_inferences: int = Field(default=0, ge=0, le=32)
    detector_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    model_weights_path: str = "src/vision/weights/yolo26n.pt"
    detector_target_labels: str = "person,backpack,handbag,suitcase"
    # Auth
    # scrypt cost as a power of two. Measured on the dev box (2026-08-10): 14 -> 122 ms,
    # 15 -> 164 ms, 16 -> 333 ms. 14 sits in the 80-150 ms band — real cost to an attacker,
    # no perceptible wait for a guard starting a shift.
    password_work_factor: int = Field(default=14, ge=12, le=20)
    # 12 hours is one shift; 30 minutes idle ends a session left open on an unattended
    # terminal. Whichever fires first wins — see src/services/sessions.py.
    session_ttl_hours: float = Field(default=12.0, gt=0.0)
    session_idle_minutes: float = Field(default=30.0, gt=0.0)
    session_max_sessions: int = Field(default=200, ge=1)
    login_max_failures: int = Field(default=5, ge=1, le=100)
    login_failure_window_minutes: float = Field(default=15.0, gt=0.0)
    login_lockout_minutes: float = Field(default=15.0, gt=0.0)
    # Deliberately off by default. A browser silently refuses a `Secure` cookie sent over
    # plain HTTP, so turning this on while the backend is on http://localhost:8000 produces
    # a login screen that accepts the password and then bounces straight back to itself with
    # no error anywhere. Turn it on only once TLS actually terminates in front of the app.
    session_cookie_secure: bool = False

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Optional distributed event relay. Empty keeps the single-process in-memory fan-out.
    redis_url: str = ""
    redis_event_channel: str = "p091.events"

    # Incident evidence is local for development and S3-compatible in production.
    evidence_storage_backend: str = "local"
    evidence_storage_dir: str = "./data"
    evidence_s3_bucket: str = ""
    evidence_s3_endpoint_url: str = ""
    evidence_s3_region: str = ""
    evidence_s3_public_base_url: str = ""

    # Semantic search is asynchronous and disabled until a compatible embedding worker is configured.
    embeddings_enabled: bool = False
    embedding_dimensions: int = Field(default=1536, ge=1, le=4096)
    embedding_model: str = "text-embedding-3-small"

    # Telegram notifications. Disabled until a bot token and chat IDs exist.
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    # Comma-separated user or group chat IDs. Telegram accepts both in the same list.
    telegram_chat_ids: str = ""
    notification_max_retries: int = Field(default=3, ge=0, le=10)
    notification_backoff_sec: float = Field(default=2.0, gt=0.0, le=60.0)

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()

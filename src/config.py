from functools import lru_cache

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

    # General LLM agent. Qwen Model Studio's OpenAI-compatible endpoint is the default,
    # but these names are provider-neutral so a deployment can point elsewhere if needed.
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3.7-flash"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # VLM triage (PRD §1.2: LLM context reasoning to filter false alarms).
    # Uses an OpenAI-compatible protocol, so the monitoring pipeline is not tied to OpenAI.
    vlm_enabled: bool = False
    vlm_api_key: str = ""
    vlm_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    vlm_model: str = "qwen3.7-flash"
    # Pre-roll must reach back past RuleEngine.fall_dwell_sec (3.0) plus the 0.5s velocity
    # window. Anything shorter and the "before" frames show a person already on the ground,
    # so the model sees no fall and correctly calls a real fall a false positive.
    vlm_pre_roll_sec: float = Field(default=5.0, gt=0.0)
    vlm_post_roll_sec: float = Field(default=2.0, ge=0.0)
    # Every frame is a billed image and accuracy degrades with long image sequences.
    vlm_max_frames: int = Field(default=6, ge=2, le=16)
    vlm_timeout_sec: float = Field(default=20.0, gt=0.0)
    vlm_history_examples: int = Field(default=6, ge=0, le=32)

    # LLM reasoning stage (Vision -> VLM -> LLM, docs/plans/llm-reasoning-node.md). Off by
    # default and layered strictly on top of vlm_enabled: with vlm_enabled=false this flag
    # does nothing, since CascadingTriage always needs a working VLM verdict to reason over.
    # Reuses llm_api_key/llm_base_url/model_name above rather than adding a third key.
    llm_reasoning_enabled: bool = False
    # Text-only call, so it should be faster than the VLM's own vlm_timeout_sec.
    llm_reasoning_timeout_sec: float = Field(default=15.0, gt=0.0)

    # Monitoring
    # Per (camera, label) floor between alerts. One flickering track would otherwise
    # open a modal on every frame it flickers.
    alert_cooldown_sec: float = Field(default=30.0, ge=0.0)
    camera_target_fps: float = Field(default=8.0, gt=0.0, le=60.0)

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
    # Deliberately off by default. A browser silently refuses a `Secure` cookie sent over
    # plain HTTP, so turning this on while the backend is on http://localhost:8000 produces
    # a login screen that accepts the password and then bounces straight back to itself with
    # no error anywhere. Turn it on only once TLS actually terminates in front of the app.
    session_cookie_secure: bool = False

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()

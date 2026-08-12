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

    # LLM
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # VLM triage (PRD §1.2: LLM context reasoning to filter false alarms)
    # Off by default, and an empty openai_api_key forces the null client regardless, so a
    # fresh clone and CI both behave exactly as they did before this layer existed.
    vlm_enabled: bool = False
    vlm_model: str = "gpt-4o-mini"
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

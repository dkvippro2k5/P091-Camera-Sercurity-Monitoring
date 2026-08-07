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

    # Monitoring
    # Per (camera, label) floor between alerts. One flickering track would otherwise
    # open a modal on every frame it flickers.
    alert_cooldown_sec: float = Field(default=30.0, ge=0.0)
    camera_target_fps: float = Field(default=8.0, gt=0.0, le=60.0)

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()

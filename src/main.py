import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.vision_routes import router as monitoring_router
from src.config import get_settings
from src.services.monitoring import MonitoringService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    print(f"Starting {settings.app_name} in {settings.app_env} mode")

    # One MonitoringService per process — it owns camera threads and YOLO weights, so it
    # cannot be built per request. Constructing it loads the detector, which takes a few
    # seconds; doing that here rather than on the first request keeps the first camera
    # from timing out.
    service = await asyncio.to_thread(MonitoringService)
    # Worker threads publish SSE events through this loop, so it must be handed over
    # before any camera starts.
    service.bind_loop(asyncio.get_running_loop())
    app.state.monitoring = service
    print(f"Monitoring service ready ({len(service.list_cameras())} camera(s) restored)")

    try:
        yield
    finally:
        print("Shutting down...")
        service.shutdown()


app = FastAPI(
    title="AI20K Agent",
    description="AI Agent built with LangGraph + 24/7 multi-camera incident monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth_routes import router as auth_router
from src.api.routes import router
from src.api.vision_routes import manager_router
from src.api.vision_routes import router as monitoring_router
from src.config import get_settings
from src.services.monitoring import MonitoringService
from src.services.sessions import SessionStore
from src.services.user_store import UserStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    print(f"Starting {settings.app_name} in {settings.app_env} mode")

    # Accounts and sessions, both process singletons like the monitoring service below.
    # Built before it, and cheaply, so that if the roster is missing the log says so before
    # anything spends seconds loading YOLO weights.
    app.state.users = UserStore()
    app.state.sessions = SessionStore(
        ttl_hours=settings.session_ttl_hours,
        idle_minutes=settings.session_idle_minutes,
        max_sessions=settings.session_max_sessions,
    )
    if not app.state.users.list_users():
        print("! No accounts exist yet — run: python scripts/seed_users.py")

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
# Public: login is what everything else is gated behind.
app.include_router(auth_router, prefix="/api/v1")
# Guard surface, then the manager-only routes. The two never share a path, so a request can
# only ever resolve against the gate it belongs to.
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(manager_router, prefix="/api/v1")


@app.get("/health")
async def health():
    """Deliberately public — it is the Docker healthcheck, which carries no cookie."""
    return {"status": "ok", "env": settings.app_env}

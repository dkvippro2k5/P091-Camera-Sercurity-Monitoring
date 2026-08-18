from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.main import app

# The lowest work factor scrypt accepts that is still a real scrypt call: 2**12 rather than
# the production 2**14, because the fixtures below hash two passwords per test and the
# production factor is ~122 ms each. Nothing under test depends on the cost — `security.py`
# stores the parameters inside each record, so a hash made at 12 verifies at 12.
TEST_WORK_FACTOR = 12


@pytest_asyncio.fixture
async def client():
    """Async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_llm():
    """Mock LLM to avoid calling OpenAI during tests.

    Usage in test:
        def test_something(mock_llm):
            # LLM calls will return mock response instead of hitting OpenAI
            ...
    """
    mock = AsyncMock()
    mock.ainvoke.return_value = AsyncMock(content="Mocked LLM response")
    return mock


# ---------------------------------------------------------------- isolation
@pytest.fixture
def isolated_data(monkeypatch, tmp_path):
    """Redirect every file a MonitoringService writes into `tmp_path`, and null the detector.

    Both JSONL logs are patched **at the class name**, not at the path constant. Their
    constructors take `path: str = DEFAULT_..._PATH`, and a default argument is bound when the
    function is defined — so `monkeypatch.setattr(module, "DEFAULT_INCIDENT_LOG_PATH", ...)`
    changes the constant and nothing else, which looks like isolation while every write still
    lands in the real `data/`.

    That is not a theoretical tidiness point. `_emit_alert` appends to `incidents.jsonl` on
    every alert, and `test_alert_ring_buffer_is_bounded` fires 210 of them; the verdict path
    appends to `vlm_history.jsonl`, which is the labelled dataset the PRD §4 precision figure
    is computed from. Unisolated, the suite would inject hundreds of synthetic rows into a
    developer's real evaluation data and quietly bias the report it exists to produce.

    `eval_report.AlertHistory` is patched separately because `build_report()` constructs its
    own instance rather than taking the service's.
    """
    from src.services import eval_report as eval_report_module
    from src.services import monitoring as monitoring_module
    from src.services.alert_history import AlertHistory
    from src.services.incident_log import IncidentLog
    from src.vision.detector import IncidentDetector

    history_path = str(tmp_path / "vlm_history.jsonl")
    monkeypatch.setattr(monitoring_module, "IncidentDetector", lambda **_: IncidentDetector(model_path=None))
    monkeypatch.setattr(monitoring_module, "REGISTRY_PATH", str(tmp_path / "cameras.json"))
    monkeypatch.setattr(monitoring_module, "AlertHistory", lambda: AlertHistory(path=history_path))
    monkeypatch.setattr(monitoring_module, "IncidentLog", lambda: IncidentLog(path=str(tmp_path / "incidents.jsonl")))
    monkeypatch.setattr(eval_report_module, "AlertHistory", lambda: AlertHistory(path=history_path))
    return tmp_path


# ---------------------------------------------------------------- auth fixtures
@pytest.fixture
def auth_app(monkeypatch, tmp_path, isolated_data):
    """The real app with an isolated roster, sessions file, camera registry and null detector.

    Every path the app writes to is redirected into `tmp_path`. That matters more than usual
    here: a test that fell through to the default paths would append fake accounts to a
    developer's real `data/users.json` and evict their live session out of `sessions.json`.

    Yields `(TestClient, UserStore)` so a test can add an account or flip `active` directly.
    """
    from src.services.sessions import SessionStore
    from src.services.user_store import UserStore

    users = UserStore(path=str(tmp_path / "users.json"))
    sessions = SessionStore(path=str(tmp_path / "sessions.json"))

    # Patched at the module the lifespan imports them from, so the singletons it builds are
    # these — pre-seeded and pointed at tmp_path — rather than ones reading the real data dir.
    import src.main as main_module

    monkeypatch.setattr(main_module, "UserStore", lambda: users)
    monkeypatch.setattr(main_module, "SessionStore", lambda **_: sessions)

    users.add_user("gwen", "guard-password", "guard", "Gwen the Guard", work_factor=TEST_WORK_FACTOR)
    users.add_user("morgan", "manager-password", "manager", "Morgan the Manager", work_factor=TEST_WORK_FACTOR)

    with TestClient(app) as test_client:
        yield test_client, users


@pytest.fixture
def anon_client(auth_app):
    """Signed out. Used to assert every gated route is a 401 rather than a 200 or a 500."""
    test_client, _ = auth_app
    return test_client


@pytest.fixture
def guard_client(auth_app):
    """Signed in as a guard. `TestClient`'s cookie jar carries the session from here on."""
    test_client, _ = auth_app
    response = test_client.post("/api/v1/auth/login", json={"username": "gwen", "password": "guard-password"})
    assert response.status_code == 200, response.text
    return test_client


@pytest.fixture
def manager_client(auth_app):
    """Signed in as a manager."""
    test_client, _ = auth_app
    response = test_client.post("/api/v1/auth/login", json={"username": "morgan", "password": "manager-password"})
    assert response.status_code == 200, response.text
    return test_client

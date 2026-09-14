import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from retired_main import HUGGINGCHAT_URL, app  # noqa: E402

client = TestClient(app)


def test_retirement_status_points_to_huggingchat():
    response = client.get("/api")

    assert response.status_code == 200
    assert response.json() == {
        "name": "ML Intern",
        "status": "retired",
        "moved_to": HUGGINGCHAT_URL,
    }


@pytest.mark.parametrize(
    "path",
    [
        "/api/session",
        "/api/health/llm",
        "/auth/login",
        "/auth/status",
    ],
)
def test_retired_server_does_not_expose_application_routes(path):
    assert client.get(path).status_code == 404

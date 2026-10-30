from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_reports_application_state_without_exposing_secrets() -> None:
    app = create_app(Settings(_env_file=None, google_api_key="test-secret"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "provider_configured": True,
    }
    assert "test-secret" not in response.text

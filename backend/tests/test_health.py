from fastapi.testclient import TestClient

from backend.app import __version__
from backend.app.main import app


client = TestClient(app)


def test_health_endpoint_returns_service_status() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "enterprise-voice-ai-agent",
        "version": __version__,
    }


def test_health_endpoint_is_documented_in_openapi() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    health_operation = response.json()["paths"]["/api/v1/health"]["get"]
    assert health_operation["tags"] == ["health"]
    assert health_operation["summary"] == "Check API health"

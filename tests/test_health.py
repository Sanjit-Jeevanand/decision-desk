"""Smoke test — verifies the app starts and health endpoint responds."""

from fastapi.testclient import TestClient
from decisiondesk.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

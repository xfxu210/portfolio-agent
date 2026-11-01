import importlib
from pathlib import Path

from fastapi.testclient import TestClient

api_module = importlib.import_module("portfolio_agent.api.app")
api_module.RUNS_ROOT = Path("runs").resolve()


def test_health() -> None:

    response = TestClient(api_module.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_id_resolves_independently_of_output_folder_name() -> None:
    client = TestClient(api_module.app)
    runs = client.get("/api/runs").json()
    if not runs:
        return
    run_id = runs[0]["run_id"]
    response = client.get(f"/api/runs/{run_id}/months")
    assert response.status_code == 200
    assert len(response.json()) > 0

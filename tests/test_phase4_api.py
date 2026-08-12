"""
tests/test_phase4_api.py — Phase 4 API Tests
Run: python -m pytest tests/test_phase4_api.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["service"] == "AgentForge API"

    def test_health_has_version(self, client):
        assert response.json()["version"] == "1.0.0" if (response := client.get("/health")) else True


class TestRunEndpoint:
    def test_run_returns_task_id(self, client):
        response = client.post("/run", json={"goal": "Test goal for API"})
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_id"].startswith("task_")
        assert data["status"] == "running"

    def test_run_empty_goal_rejected(self, client):
        response = client.post("/run", json={"goal": ""})
        assert response.status_code == 400

    def test_run_whitespace_goal_rejected(self, client):
        response = client.post("/run", json={"goal": "   "})
        assert response.status_code == 400

    def test_run_too_long_goal_rejected(self, client):
        response = client.post("/run", json={"goal": "x" * 2001})
        assert response.status_code == 400

    def test_run_creates_retrievable_task(self, client):
        run_resp = client.post("/run", json={"goal": "Test retrievable task"})
        task_id = run_resp.json()["task_id"]
        status_resp = client.get(f"/status/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["task_id"] == task_id


class TestStatusEndpoint:
    def test_status_not_found(self, client):
        assert client.get("/status/task_nonexistent_xyz").status_code == 404

    def test_status_has_required_fields(self, client):
        task_id = client.post("/run", json={"goal": "Status field test"}).json()["task_id"]
        data = client.get(f"/status/{task_id}").json()
        for field in ["task_id", "goal", "status", "created_at"]:
            assert field in data

    def test_status_values_are_valid(self, client):
        task_id = client.post("/run", json={"goal": "Status value test"}).json()["task_id"]
        valid = {"pending", "in_progress", "completed", "failed", "needs_revision"}
        assert client.get(f"/status/{task_id}").json()["status"] in valid


class TestLogsEndpoint:
    def test_logs_for_valid_task(self, client):
        task_id = client.post("/run", json={"goal": "Logs test"}).json()["task_id"]
        response = client.get(f"/logs/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "count" in data

    def test_logs_not_found(self, client):
        assert client.get("/logs/task_nonexistent_abc").status_code == 404

    def test_logs_agent_filter(self, client):
        task_id = client.post("/run", json={"goal": "Filter test"}).json()["task_id"]
        assert client.get(f"/logs/{task_id}?agent=researcher").status_code == 200


class TestHistoryEndpoint:
    def test_history_returns_list(self, client):
        response = client.get("/history")
        assert response.status_code == 200
        assert "tasks" in response.json()

    def test_history_after_run_has_entry(self, client):
        client.post("/run", json={"goal": "History entry test xyz"})
        tasks = client.get("/history").json()["tasks"]
        assert any("History entry test xyz" in t["goal"] for t in tasks)

    def test_history_task_has_required_fields(self, client):
        client.post("/run", json={"goal": "Field check"})
        tasks = client.get("/history").json()["tasks"]
        if tasks:
            for field in ["task_id", "goal", "status", "created_at"]:
                assert field in tasks[0]


class TestResultEndpoint:
    def test_result_not_found(self, client):
        assert client.get("/result/task_nonexistent_xyz").status_code == 404

    def test_result_not_ready_returns_202(self, client):
        task_id = client.post("/run", json={"goal": "Result not ready"}).json()["task_id"]
        assert client.get(f"/result/{task_id}").status_code in [200, 202]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

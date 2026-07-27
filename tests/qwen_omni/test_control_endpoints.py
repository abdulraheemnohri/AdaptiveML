"""
Unit and integration tests for ModelServer control and single-user spec endpoints.
"""

import pytest
import torch.nn as nn
from fastapi.testclient import TestClient
from adaptive_ml.serving.inference import ModelServer


class DummyModel(nn.Module):
    """Simple mock PyTorch module for server testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x, attention_mask=None):
        return x


@pytest.fixture
def test_client():
    """Fixture providing a TestClient for ModelServer."""
    model = DummyModel()
    server = ModelServer(model=model)
    app = server.get_app()
    return TestClient(app)


def test_control_learning_cycle_endpoints(test_client):
    """Test learning cycle start, pause, and stop controls."""
    # Start learning
    resp = test_client.post("/control/start-learning")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["state"]["current_learning"] == "Continual Learning"
    assert resp.json()["state"]["status"] == "LEARNING..."

    # Pause learning
    resp = test_client.post("/control/pause-learning")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["state"]["current_learning"] == "PAUSED"
    assert resp.json()["state"]["status"] == "PAUSED"

    # Stop learning
    resp = test_client.post("/control/stop-learning")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["state"]["current_learning"] == "STOPPED"
    assert resp.json()["state"]["status"] == "STOPPED"


def test_control_task_triggers(test_client):
    """Test various training, testing, and gap discovery triggers."""
    # Test model
    resp = test_client.post("/control/test-model")
    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "TESTING..."

    # Forgetting test
    resp = test_client.post("/control/run-forgetting-test")
    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "CHECKING..."

    # Find gaps
    resp = test_client.post("/control/find-gaps")
    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "GAP DISCOVERY..."

    # Collect data
    resp = test_client.post("/control/collect-data")
    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "ACQUIRING..."

    # Train candidate
    resp = test_client.post("/control/train-candidate")
    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "TRAINING..."

    # Compare models
    resp = test_client.post("/control/compare-models")
    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "COMPARING..."


def test_registry_control_triggers(test_client):
    """Test rollback and emergency promotion."""
    # Rollback
    resp = test_client.post("/control/rollback")
    assert resp.status_code == 200
    assert "Rolled Back" in resp.json()["state"]["current_model"]
    assert resp.json()["state"]["status"] == "ROLLED BACK"

    # Emergency Promote
    resp = test_client.post("/control/emergency-promote")
    assert resp.status_code == 200
    assert "Promoted" in resp.json()["state"]["current_model"]
    assert resp.json()["state"]["status"] == "PROMOTED"


def test_conversational_chat_endpoint(test_client):
    """Test conversational chat endpoint with text, Urdu, and python keywords."""
    # Standard text query
    resp = test_client.post("/chat", json={"text": "Hello"})
    assert resp.status_code == 200
    assert "Greetings!" in resp.json()["prediction"]
    assert "Base model" in resp.json()["explained_answer"]

    # Urdu skill adapter trigger
    resp = test_client.post("/chat", json={"text": "Can you translate Urdu?"})
    assert resp.status_code == 200
    assert "language adapter" in resp.json()["prediction"]
    assert "Urdu Skill Adapter" in resp.json()["explained_answer"]

    # Coding adapter trigger
    resp = test_client.post("/chat", json={"text": "Write some python code"})
    assert resp.status_code == 200
    assert "def compute_fisher_information" in resp.json()["prediction"]
    assert "Coding Adapter" in resp.json()["explained_answer"]


def test_long_term_memory_endpoints(test_client):
    """Test listing, adding, toggling trust, and forgetting memories."""
    # List memories
    resp = test_client.get("/memory")
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) >= 1

    # Add memory
    new_mem = {"type": "user", "content": "Prefers light mode style."}
    resp = test_client.post("/memory", json=new_mem)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    added_id = resp.json()["memory"]["id"]

    # Toggle trust
    resp = test_client.post(f"/memory/{added_id}/trust")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Delete memory
    resp = test_client.delete(f"/memory/{added_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_human_feedback_endpoint(test_client):
    """Test submitting human feedback."""
    feedback = {
        "message_id": "msg-123",
        "rating": 5,
        "is_hallucination": False,
        "is_factual_error": False,
        "correction": "The code works perfectly."
    }
    resp = test_client.post("/feedback", json=feedback)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_data_sources_endpoints(test_client):
    """Test listing, adding, and testing data sources."""
    # List
    resp = test_client.get("/data/sources")
    assert resp.status_code == 200
    assert len(resp.json()["sources"]) >= 1
    src_id = resp.json()["sources"][0]["id"]

    # Add
    new_src = {"name": "Test Local CSV", "type": "csv", "priority": "low", "trust_level": "trusted"}
    resp = test_client.post("/data/sources", json=new_src)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Test
    resp = test_client.post(f"/data/sources/{src_id}/test")
    assert resp.status_code == 200
    assert "passed" in resp.json()["message"]


def test_knowledge_gaps_endpoints(test_client):
    """Test listing and adding gaps."""
    # List
    resp = test_client.get("/gaps")
    assert resp.status_code == 200
    assert len(resp.json()["gaps"]) >= 1

    # Add
    new_gap = {"topic": "MMM-Speech Audio patterns", "importance": "medium", "confidence": "low"}
    resp = test_client.post("/gaps", json=new_gap)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_autonomous_agents_endpoints(test_client):
    """Test listing and toggling agents autonomy level."""
    # List
    resp = test_client.get("/agents")
    assert resp.status_code == 200
    assert len(resp.json()["agents"]) >= 1
    agent_id = resp.json()["agents"][0]["id"]

    # Toggle
    resp = test_client.post(f"/agents/{agent_id}/toggle", json={"autonomous_level": "autonomous"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["agent"]["autonomous_level"] == "autonomous"


def test_system_alerts_endpoints(test_client):
    """Test listing and clearing alerts."""
    # List
    resp = test_client.get("/alerts")
    assert resp.status_code == 200
    assert len(resp.json()["alerts"]) >= 1

    # Clear
    resp = test_client.post("/alerts/clear")
    assert resp.status_code == 200
    assert len(test_client.get("/alerts").json()["alerts"]) == 0

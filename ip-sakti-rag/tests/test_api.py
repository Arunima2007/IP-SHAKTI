"""API Integration Tests for IP-SAKTI Sahayak (Milestone 6)."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Test client fixture with lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifies root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "IP-SAKTI Sahayak API"
    assert "version" in data


def test_health_endpoint(client):
    """Verifies health endpoint returns chunk counts and model statuses."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend_connected"] is True
    assert data["total_chunks_indexed"] >= 5000
    assert data["total_documents"] == 22
    assert "LangGraph" in data["orchestrator"]


def test_documents_endpoint(client):
    """Verifies document catalog endpoint returns all 22 authoritative sources."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 22
    assert data["total_chunks"] >= 5000
    assert len(data["categories"]) >= 4
    assert len(data["documents"]) == 22
    
    # Check specific documents exist
    doc_ids = [d["id"] for d in data["documents"]]
    assert "Patent Act-1970.pdf" in doc_ids
    assert "The Biological Diversity Act,2002.pdf" in doc_ids
    assert "Drugs_and_Cosmetics_Act_1940.pdf" in doc_ids


def test_chat_empty_query(client):
    """Verifies empty query rejection."""
    response = client.post("/api/chat", json={"query": "   ", "language": "auto"})
    assert response.status_code in [400, 422]


def test_chat_out_of_scope_query(client):
    """Verifies safe refusal for out-of-scope query."""
    response = client.post(
        "/api/chat",
        json={"query": "Who will win the IPL cricket match tomorrow?", "language": "auto"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert "outside the scope of IP-SAKTI Sahayak" in data["answer"]
    assert data["query_type"] == "OUT_OF_SCOPE"


def test_chat_in_scope_exact_lookup(client):
    """Verifies in-scope query returns grounded answer with structured citations."""
    response = client.post(
        "/api/chat",
        json={
            "query": "What does Section 3(p) of the Indian Patents Act, 1970 state?",
            "language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is False
    assert len(data["citations"]) >= 1
    assert data["validation"]["status"] == "VALID"
    assert "Patents Act" in data["answer"]
    assert data["metadata"]["latency_ms"] > 0

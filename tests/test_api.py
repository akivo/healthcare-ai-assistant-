"""
test_api.py — API Endpoint Tests
==================================
These are integration tests that test the full API endpoints.
They use FastAPI's built-in TestClient which simulates HTTP requests
without needing a running server.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def client():
    """Create a test client for our FastAPI app."""
    # We patch Ollama and ChromaDB so tests don't need them running
    with patch('app.llm.check_ollama_health') as mock_health, \
         patch('app.embeddings.get_embedding_model') as mock_embed:
        
        mock_health.return_value = {"status": "connected", "model": "phi3:mini"}
        mock_embed.return_value = MagicMock()
        
        from app.main import app
        return TestClient(app)


def test_health_endpoint(client):
    """Test that GET /health returns 200 with expected fields."""
    with patch('app.main.check_ollama_health') as mock_health, \
         patch('app.main.get_vector_store') as mock_vs:
        
        mock_health.return_value = {"status": "connected", "model": "phi3:mini"}
        mock_vs.return_value.get.return_value = {"ids": ["id1", "id2"]}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "llm" in data
        assert "vector_store" in data


def test_ask_endpoint_valid_question(client):
    """Test that POST /ask with a valid question returns expected response format."""
    mock_response = {
        "answer": "Yes, you can request medication refills through telehealth.",
        "sources": [{"document": "telehealth.txt", "chunk": "Refills can be done...", "relevance_score": 0.85}],
        "confidence": "high",
        "route": "rag",
    }
    
    with patch('app.main.process_question', return_value=mock_response):
        response = client.post("/ask", json={"question": "Can I get a refill via telehealth?"})
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "route" in data
        assert data["confidence"] in ["high", "medium", "low"]


def test_ask_endpoint_empty_question(client):
    """Test that POST /ask with empty question returns 422 validation error."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422  # Unprocessable Entity


def test_ask_endpoint_missing_field(client):
    """Test that POST /ask with missing 'question' field returns 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ingest_endpoint(client):
    """Test that POST /ingest calls the ingestion pipeline."""
    mock_result = {
        "status": "success",
        "documents_processed": 6,
        "chunks_created": 45,
        "chunks_stored": 45,
        "source_files": ["doc1.txt", "doc2.txt"],
        "vector_store": "chromadb",
        "message": "Successfully ingested 6 documents.",
    }
    
    with patch('app.main.ingest_documents', return_value=mock_result):
        response = client.post("/ingest", json={"data_dir": "./data"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_processed"] == 6


def test_ask_endpoint_appointment_routing(client):
    """Test that appointment questions get routed to appointment tool."""
    mock_response = {
        "answer": "Available slots for Cardiology on Monday: 9:00 AM, 11:30 AM",
        "sources": [{"document": "mock_appointment_system", "chunk": "...", "relevance_score": 1.0}],
        "confidence": "high",
        "route": "appointment_tool",
    }
    
    with patch('app.main.process_question', return_value=mock_response):
        response = client.post("/ask", json={"question": "Book a cardiology appointment for Monday"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["route"] == "appointment_tool"

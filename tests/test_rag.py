import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_pdf_bytes() -> bytes:
    """Minimal valid PDF for upload tests."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n9\n%%EOF"
    )


def test_upload_pdf_success(client):
    mock_result = {"filename": "test.pdf", "pages": 1, "chunks": 3}
    with patch("app.main.index_pdf", new_callable=AsyncMock, return_value=mock_result):
        response = client.post(
            "/api/v1/documents",
            files={"file": ("test.pdf", _make_pdf_bytes(), "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Document indexed successfully"
    assert data["filename"] == "test.pdf"
    assert data["chunks"] == 3


def test_upload_non_pdf_rejected(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("report.txt", b"Some text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_empty_file_rejected(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_rag_query_with_results(client):
    mock_result = {
        "answer": "The document discusses climate change.",
        "sources": [{"filename": "report.pdf", "page": 2}],
        "context_used": True,
        "chunks_retrieved": 2,
        "duration_seconds": 1.2,
    }
    with patch("app.main.query_rag", new_callable=AsyncMock, return_value=mock_result):
        response = client.post(
            "/api/v1/rag/query",
            json={"question": "What does the document discuss?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["context_used"] is True
    assert len(data["sources"]) == 1
    assert "climate change" in data["answer"]


def test_rag_query_no_results(client):
    mock_result = {
        "answer": "No relevant documents were found in the knowledge base.",
        "sources": [],
        "context_used": False,
        "chunks_retrieved": 0,
    }
    with patch("app.main.query_rag", new_callable=AsyncMock, return_value=mock_result):
        response = client.post(
            "/api/v1/rag/query",
            json={"question": "Completely unrelated question"},
        )
    assert response.status_code == 200
    assert response.json()["context_used"] is False


def test_rag_query_empty_question(client):
    response = client.post("/api/v1/rag/query", json={"question": ""})
    assert response.status_code == 422


def test_agent_query_success(client):
    mock_result = {
        "answer": "Based on the documents, the answer is 42.",
        "tools_used": ["search_documents"],
        "steps": 3,
    }
    with patch("app.main.run_agent", new_callable=AsyncMock, return_value=mock_result):
        response = client.post(
            "/api/v1/agent/query",
            json={"query": "What is the answer according to the documents?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "tools_used" in data


def test_calculate_tool():
    from app.agent import calculate

    assert calculate.invoke({"expression": "2 + 2"}) == "4.0"
    assert calculate.invoke({"expression": "10 * 5"}) == "50.0"
    assert calculate.invoke({"expression": "2 ** 10"}) == "1024.0"
    assert calculate.invoke({"expression": "100 / 4"}) == "25.0"


def test_calculate_tool_invalid_expression():
    from app.agent import calculate

    result = calculate.invoke({"expression": "import os"})
    assert "error" in result.lower()


def test_calculate_tool_division_by_zero():
    from app.agent import calculate

    result = calculate.invoke({"expression": "1 / 0"})
    assert "error" in result.lower() or result == "inf"

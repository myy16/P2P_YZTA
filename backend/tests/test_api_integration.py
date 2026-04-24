import os
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Allow imports from backend/ without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.api.summarize import router as summarize_router


class FakeRagService:
    def answer_question(self, question, top_k=5, file_id=None, source_file=None):
        return {
            "answer": f"yanit:{question}",
            "sources": [{"source_file": source_file or "doc.pdf", "chunk_index": 0}],
            "context": [{"text": "context"}],
            "model": "test-model",
        }

    def answer_question_stream(self, question, top_k=5, file_id=None, source_file=None):
        yield 'data: {"type":"token","content":"Merhaba"}\n\n'
        yield 'data: {"type":"token","content":" dunya"}\n\n'
        yield 'data: {"type":"sources","content":[{"source_file":"doc.pdf","chunk_index":0}]}\n\n'

    def summarize_documents(self, file_id=None, source_file=None, max_chunks=8):
        return {
            "summary": "kisa ozet",
            "sources": [{"source_file": source_file or "doc.pdf", "chunk_index": 0}],
            "context": [{"text": "context"}],
            "model": "test-model",
        }


def create_test_client():
    app = FastAPI()
    app.include_router(upload_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(summarize_router, prefix="/api")
    return TestClient(app)


def test_chat_endpoint_returns_answer(monkeypatch):
    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: FakeRagService())
    client = create_test_client()

    response = client.post(
        "/api/chat",
        json={"question": "Belge ne anlatıyor?", "source_file": "rapor.pdf", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "yanit:Belge ne anlatıyor?"
    assert data["sources"][0]["source_file"] == "rapor.pdf"


def test_upload_stream_emits_stage_and_done_events(monkeypatch):
    monkeypatch.setattr("app.api.upload.parse_document", lambda path, ext: "Temiz metin")
    monkeypatch.setattr("app.api.upload.clean_text", lambda text: text)
    monkeypatch.setattr(
        "app.api.upload.chunk_text",
        lambda text, metadata: [{"chunk_id": "1", "text": text, "metadata": metadata, "chunk_index": 0, "source_file": metadata["source_file"], "file_id": metadata["file_id"]}],
    )

    class UploadService:
        def __init__(self):
            self.vector_store = type("Store", (), {"fetch_all": lambda self, filters=None: {"ids": []}, "collection": lambda self: type("Col", (), {"delete": lambda self, ids=None: None})()})()

        def index_chunks(self, chunks):
            return len(chunks)

    monkeypatch.setattr("app.api.upload.get_rag_service", lambda: UploadService())
    client = create_test_client()

    response = client.post(
        "/api/upload/stream",
        files={"files": ("rapor.txt", b"merhaba dunya", "text/plain")},
        data={"username": "test-user"},
    )

    assert response.status_code == 200
    body = response.text
    assert '"event": "stage"' in body
    assert '"stage": "Metin çıkarılıyor"' in body
    assert '"event": "file_done"' in body
    assert '"event": "done"' in body


def test_chat_endpoint_maps_runtime_error_to_503(monkeypatch):
    class BrokenService:
        def answer_question(self, *args, **kwargs):
            raise RuntimeError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: BrokenService())
    client = create_test_client()

    response = client.post("/api/chat", json={"question": "test"})

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_chat_endpoint_maps_unexpected_error_to_500(monkeypatch):
    class BrokenService:
        def answer_question(self, *args, **kwargs):
            raise ValueError("unexpected")

    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: BrokenService())
    client = create_test_client()

    response = client.post("/api/chat", json={"question": "test"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."


def test_chat_stream_endpoint_returns_sse(monkeypatch):
    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: FakeRagService())
    client = create_test_client()

    response = client.post("/api/chat/stream", json={"question": "stream deneme"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type":"token"' in body
    assert '"type":"sources"' in body


def test_chat_stream_endpoint_emits_error_event_on_runtime_error(monkeypatch):
    class BrokenService:
        def answer_question_stream(self, *args, **kwargs):
            raise RuntimeError("stream unavailable")
            yield  # pragma: no cover

    monkeypatch.setattr("app.api.chat.get_rag_service", lambda: BrokenService())
    client = create_test_client()

    response = client.post("/api/chat/stream", json={"question": "stream deneme"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "error"' in response.text
    assert 'stream unavailable' in response.text


def test_summarize_endpoint_returns_summary(monkeypatch):
    monkeypatch.setattr("app.api.summarize.get_rag_service", lambda: FakeRagService())
    client = create_test_client()

    response = client.post(
        "/api/summarize",
        json={"source_file": "rapor.pdf", "max_chunks": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "kisa ozet"
    assert data["sources"][0]["source_file"] == "rapor.pdf"


def test_summarize_endpoint_maps_runtime_error_to_503(monkeypatch):
    class BrokenService:
        def summarize_documents(self, *args, **kwargs):
            raise RuntimeError("service temporarily unavailable")

    monkeypatch.setattr("app.api.summarize.get_rag_service", lambda: BrokenService())
    client = create_test_client()

    response = client.post("/api/summarize", json={"source_file": "doc.pdf"})

    assert response.status_code == 503
    assert response.json()["detail"] == "service temporarily unavailable"


def test_summarize_endpoint_maps_unexpected_error_to_500(monkeypatch):
    class BrokenService:
        def summarize_documents(self, *args, **kwargs):
            raise ValueError("unexpected")

    monkeypatch.setattr("app.api.summarize.get_rag_service", lambda: BrokenService())
    client = create_test_client()

    response = client.post("/api/summarize", json={"source_file": "doc.pdf"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."

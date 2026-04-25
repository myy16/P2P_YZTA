import os
import sys

# Allow imports from backend/ without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.retriever import Retriever


class FakeEmbeddingService:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def query(self, query_embedding, top_k=5, filters=None):
        self.calls.append({"top_k": top_k, "filters": filters})

        # First pass: weak lexical match, should trigger low recall.
        if len(self.calls) == 1:
            return {
                "documents": [["hava durumu ve bulutlar", "spor haberi"]],
                "metadatas": [[
                    {"source_file": "weather.txt", "chunk_index": 0, "file_id": "f1"},
                    {"source_file": "sport.txt", "chunk_index": 0, "file_id": "f2"},
                ]],
                "distances": [[0.15, 0.18]],
            }

        # Second pass: contains lexical overlap with query.
        return {
            "documents": [["finansal rapor gelir artisi ve net kar", "ekonomi raporu gider analizi"]],
            "metadatas": [[
                {"source_file": "finance.txt", "chunk_index": 1, "file_id": "f3"},
                {"source_file": "economy.txt", "chunk_index": 2, "file_id": "f4"},
            ]],
            "distances": [[0.45, 0.55]],
        }


def _build_retriever(monkeypatch):
    monkeypatch.setattr("app.core.retriever.get_embedding_service", lambda: FakeEmbeddingService())
    monkeypatch.setattr("app.core.retriever.get_vector_store", lambda: FakeVectorStore())
    return Retriever()


def test_retriever_runs_second_pass_on_low_recall(monkeypatch):
    retriever = _build_retriever(monkeypatch)

    diagnostics = retriever.retrieve_with_diagnostics("raporda gelir bilgisi nedir", top_k=3, username="u1")
    results = diagnostics["chunks"]

    assert len(retriever.vector_store.calls) >= 2
    assert any(item.get("source_file") == "finance.txt" for item in results)
    assert "confidence_score" in diagnostics
    assert "context_coverage" in diagnostics
    assert "retrieval_quality" in diagnostics


def test_retriever_uses_dynamic_top_k_and_hybrid_scoring(monkeypatch):
    retriever = _build_retriever(monkeypatch)

    results = retriever.retrieve("detayli finansal raporda gelir artis oranlari nelerdir", top_k=2)

    # Dynamic top-k should request a larger candidate pool than plain top_k.
    assert retriever.vector_store.calls[0]["top_k"] > 2
    assert results
    assert "combined_score" in results[0]
    assert "semantic_score" in results[0]
    assert "lexical_score" in results[0]
    assert "rrf_score" in results[0]
    assert "final_score" in results[0]

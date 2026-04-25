from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.core.config import (
    CHROMA_TOP_K,
    RETRIEVER_MAX_CANDIDATE_K,
    RETRIEVER_MAX_DYNAMIC_TOP_K,
    RETRIEVER_MIN_CONFIDENCE_SCORE,
    RETRIEVER_MIN_CONTEXT_RECALL,
    RETRIEVER_MIN_RELEVANCE_SCORE,
    RETRIEVER_RRF_K,
)
from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store


class Retriever:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.min_relevance_score = RETRIEVER_MIN_RELEVANCE_SCORE
        self.min_context_recall = RETRIEVER_MIN_CONTEXT_RECALL
        self.min_confidence_score = RETRIEVER_MIN_CONFIDENCE_SCORE
        self.max_dynamic_top_k = RETRIEVER_MAX_DYNAMIC_TOP_K
        self.max_candidate_k = RETRIEVER_MAX_CANDIDATE_K
        self.rrf_k = RETRIEVER_RRF_K

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+", text.lower())

    def _dynamic_top_k(self, query: str, requested_top_k: int) -> int:
        token_count = len(self._tokenize(query))
        boost = 2 if token_count >= 8 else (1 if token_count >= 4 else 0)
        dynamic = requested_top_k + boost
        return max(1, min(dynamic, self.max_dynamic_top_k))

    def _query_variants(self, query: str) -> List[str]:
        tokens = self._tokenize(query)
        keyword_tokens = []
        for token in tokens:
            if len(token) <= 2:
                continue
            if token in {"ve", "veya", "ile", "icin", "için", "bir", "bu", "şu", "su", "mi", "mu", "mı", "mü"}:
                continue
            keyword_tokens.append(token)

        variants = [query]
        if keyword_tokens:
            variants.append(" ".join(keyword_tokens[:8]))
            variants.append(" ".join(sorted(keyword_tokens, key=len, reverse=True)[:5]))

        seen = set()
        unique = []
        for item in variants:
            norm = " ".join(item.split())
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(norm)
        return unique

    @staticmethod
    def _semantic_similarity(distance: Optional[float]) -> float:
        if distance is None:
            return 0.0
        value = float(distance)
        # If distance is cosine-like (0 best), convert to cosine similarity proxy.
        if 0.0 <= value <= 2.0:
            return max(0.0, 1.0 - value)
        return 1.0 / (1.0 + max(0.0, value))

    @staticmethod
    def _lexical_overlap(query_tokens: List[str], text_tokens: List[str]) -> float:
        if not query_tokens or not text_tokens:
            return 0.0
            
        # Master Level: Robust Prefix Matching (handle Turkish suffixes efficiently)
        # We look at the first 4-5 chars to bridge 'Yusuf' and 'Yusufun'
        def get_normalized_prefixes(tokens):
            return {t[:5].lower().strip() for t in tokens if len(t) >= 2}
            
        q_prefixes = get_normalized_prefixes(query_tokens)
        t_prefixes = get_normalized_prefixes(text_tokens)
        
        if not q_prefixes:
            return 0.0
            
        overlap = len(q_prefixes & t_prefixes)
        return float(overlap / len(q_prefixes))

    def _score_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            text = candidate.get("text", "")
            lexical = self._lexical_overlap(query_tokens, self._tokenize(text))
            semantic = self._semantic_similarity(candidate.get("distance"))
            combined = (0.7 * semantic) + (0.3 * lexical)
            enriched = dict(candidate)
            enriched["lexical_score"] = lexical
            enriched["semantic_score"] = semantic
            enriched["combined_score"] = combined
            scored.append(enriched)

        scored.sort(key=lambda item: item.get("combined_score", 0.0), reverse=True)
        return scored

    @staticmethod
    def _candidate_key(candidate: Dict[str, Any]) -> tuple:
        return (
            candidate.get("file_id"),
            candidate.get("source_file"),
            candidate.get("chunk_index"),
        )

    def _rrf_fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> Dict[tuple, float]:
        fused: Dict[tuple, float] = {}
        for ranked in ranked_lists:
            for rank, candidate in enumerate(ranked, start=1):
                key = self._candidate_key(candidate)
                fused[key] = fused.get(key, 0.0) + (1.0 / (self.rrf_k + rank))
        return fused

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(Retriever._tokenize(text))

    def _noise_filter(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        seen_texts = set()
        query_tokens = self._tokenize(query)

        for candidate in candidates:
            text = candidate.get("text", "") or ""
            normalized = self._normalized_text(text)
            if len(normalized) < 5:
                continue
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            lexical = candidate.get("lexical_score", 0.0)
            combined = candidate.get("combined_score", 0.0)
            
            # Master Level: Adaptive Noise Filtering
            # If we have some lexical match (proper noun etc), we keep it even if semantic is low
            if lexical >= 0.20:
                filtered.append(candidate)
            elif combined >= self.min_relevance_score:
                filtered.append(candidate)
            elif len(query_tokens) <= 3 and combined >= (self.min_relevance_score * 0.7):
                # Be more lenient with very short, specific queries
                filtered.append(candidate)

        return filtered

    def _estimate_confidence(self, selected: List[Dict[str, Any]], context_coverage: float) -> float:
        if not selected:
            return 0.0
        avg_semantic = sum(item.get("semantic_score", 0.0) for item in selected) / len(selected)
        avg_rrf = sum(item.get("rrf_score", 0.0) for item in selected) / len(selected)
        confidence = (0.45 * avg_semantic) + (0.35 * avg_rrf) + (0.20 * context_coverage)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _estimate_context_recall(query: str, candidates: List[Dict[str, Any]]) -> float:
        query_tokens = set(Retriever._tokenize(query))
        if not query_tokens:
            return 1.0
        context_tokens: set = set()
        for candidate in candidates:
            context_tokens |= set(Retriever._tokenize(candidate.get("text", "")))
        return len(query_tokens & context_tokens) / max(1, len(query_tokens))

    @staticmethod
    def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for candidate in candidates:
            key = (
                candidate.get("file_id"),
                candidate.get("chunk_index"),
                candidate.get("source_file"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @staticmethod
    def _build_filters(
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        conditions = []
        if file_id:
            conditions.append({"file_id": {"$eq": file_id}})
        if source_file:
            conditions.append({"source_file": {"$eq": source_file}})
        if username:
            conditions.append({"username": {"$eq": username}})
        if len(conditions) == 1:
            return conditions[0]
        if len(conditions) > 1:
            return {"$and": conditions}
        return {}

    def retrieve(
        self,
        query: str,
        top_k: int = CHROMA_TOP_K,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        diagnostics = self.retrieve_with_diagnostics(
            query=query,
            top_k=top_k,
            file_id=file_id,
            source_file=source_file,
            username=username,
        )
        return diagnostics.get("chunks", [])

    def retrieve_with_diagnostics(
        self,
        query: str,
        top_k: int = CHROMA_TOP_K,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_embedding = self.embedding_service.embed_query(query)
        if not query_embedding:
            return {
                "chunks": [],
                "confidence_score": 0.0,
                "context_coverage": 0.0,
                "retrieval_quality": False,
                "query_variants": [],
            }

        effective_top_k = self._dynamic_top_k(query, top_k)
        candidate_k = min(self.max_candidate_k, max(effective_top_k * 4, effective_top_k + 2))
        filters = self._build_filters(file_id=file_id, source_file=source_file, username=username)

        variants = self._query_variants(query)
        ranked_per_variant: List[List[Dict[str, Any]]] = []
        key_to_candidate: Dict[tuple, Dict[str, Any]] = {}

        for variant in variants:
            variant_embedding = self.embedding_service.embed_query(variant)
            if not variant_embedding:
                continue
            raw = self.vector_store.query(query_embedding=variant_embedding, top_k=candidate_k, filters=filters or None)
            scored = self._score_candidates(query, self._format_results(raw))
            ranked_per_variant.append(scored)
            for candidate in scored:
                key = self._candidate_key(candidate)
                if key not in key_to_candidate or candidate.get("combined_score", 0.0) > key_to_candidate[key].get("combined_score", 0.0):
                    key_to_candidate[key] = candidate

        fused_scores = self._rrf_fuse(ranked_per_variant)
        max_rrf = max(fused_scores.values()) if fused_scores else 1.0

        merged: List[Dict[str, Any]] = []
        for key, candidate in key_to_candidate.items():
            enriched = dict(candidate)
            enriched["rrf_score"] = fused_scores.get(key, 0.0) / max_rrf if max_rrf else 0.0
            enriched["final_score"] = (0.6 * enriched.get("combined_score", 0.0)) + (0.4 * enriched.get("rrf_score", 0.0))
            merged.append(enriched)

        merged.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        cleaned = self._noise_filter(merged, query)
        selected = cleaned[:effective_top_k]
        context_coverage = self._estimate_context_recall(query, selected)

        # Additional retrieval step when context appears insufficient.
        if context_coverage < self.min_context_recall and len(selected) < max(2, effective_top_k):
            expanded_tokens = sorted(self._tokenize(query), key=len, reverse=True)[:6]
            if expanded_tokens:
                expanded_query = " ".join(expanded_tokens)
                expanded_embedding = self.embedding_service.embed_query(expanded_query)
                if expanded_embedding:
                    raw = self.vector_store.query(
                        query_embedding=expanded_embedding,
                        top_k=min(self.max_candidate_k, candidate_k * 2),
                        filters=filters or None,
                    )
                    expanded_ranked = self._score_candidates(query, self._format_results(raw))
                    ranked_per_variant.append(expanded_ranked)
                    fused_scores = self._rrf_fuse(ranked_per_variant)
                    max_rrf = max(fused_scores.values()) if fused_scores else 1.0

                    for candidate in expanded_ranked:
                        key = self._candidate_key(candidate)
                        base = key_to_candidate.get(key, candidate)
                        key_to_candidate[key] = base

                    merged = []
                    for key, candidate in key_to_candidate.items():
                        enriched = dict(candidate)
                        enriched["rrf_score"] = fused_scores.get(key, 0.0) / max_rrf if max_rrf else 0.0
                        enriched["final_score"] = (0.6 * enriched.get("combined_score", 0.0)) + (0.4 * enriched.get("rrf_score", 0.0))
                        merged.append(enriched)

                    merged.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
                    cleaned = self._noise_filter(merged, query)
                    selected = cleaned[:effective_top_k]
                    context_coverage = self._estimate_context_recall(query, selected)

        confidence = self._estimate_confidence(selected, context_coverage)
        retrieval_quality = bool(selected) and (context_coverage >= self.min_context_recall) and (confidence >= self.min_confidence_score)

        return {
            "chunks": selected,
            "confidence_score": round(confidence, 3),
            "context_coverage": round(context_coverage, 3),
            "retrieval_quality": retrieval_quality,
            "query_variants": variants,
        }

    def fetch_documents(
        self,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters = self._build_filters(file_id=file_id, source_file=source_file, username=username)
        results = self.vector_store.fetch_all(filters=filters or None)
        return self._format_fetched_documents(results)

    @staticmethod
    def _format_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted: List[Dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            formatted.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "distance": distances[index] if index < len(distances) else None,
                    "chunk_index": metadata.get("chunk_index"),
                    "source_file": metadata.get("source_file"),
                    "file_id": metadata.get("file_id"),
                }
            )
        return formatted

    @staticmethod
    def _format_fetched_documents(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        ids = results.get("ids", []) or []

        formatted: List[Dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            formatted.append(
                {
                    "id": ids[index] if index < len(ids) else None,
                    "text": text,
                    "metadata": metadata,
                    "chunk_index": metadata.get("chunk_index"),
                    "source_file": metadata.get("source_file"),
                    "file_id": metadata.get("file_id"),
                }
            )
        return formatted


def get_retriever() -> Retriever:
    return Retriever()
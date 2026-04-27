from __future__ import annotations

import re
import numpy as np
from typing import Any, Dict, List, Set, Optional

# Master Level: Local embedding-based semantic similarity
_SEMANTIC_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    # We use a very light multilingual model, strictly from local files to avoid startup hang
    _SEMANTIC_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
except Exception as exc:
    import logging
    logging.getLogger(__name__).warning("Semantic model not found locally, using Jaccard fallback: %s", exc)
    pass

def _semantic_sim(text1: str, text2: str) -> float:
    if not _SEMANTIC_MODEL or not text1 or not text2:
        return 0.0
    emb1 = _SEMANTIC_MODEL.encode(text1, convert_to_tensor=True)
    emb2 = _SEMANTIC_MODEL.encode(text2, convert_to_tensor=True)
    from sentence_transformers.util import cos_sim
    return float(cos_sim(emb1, emb2))

_STOPWORDS = {
    "ve", "veya", "ile", "icin", "için", "ama", "fakat", "gibi", "daha", "cok", "çok",
    "bir", "bu", "su", "şu", "o", "mi", "mu", "mı", "mü", "de", "da", "ki", "ne", "nasil", "nasıl",
    "ise", "ken", "mı", "mi", "mu", "mü", "miyiz", "muyuz", "mısınız", "misiniz",
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Turkish lower case mapping
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    # Keep only alphanumeric and spaces, then split
    raw_tokens = re.findall(r"[a-z0-9çğıöşü]+", text)
    
    tokens = []
    for t in raw_tokens:
        if t in _STOPWORDS or len(t) < 2:
            continue
        # Simple stemming: take first 5 chars for agglutinative languages like Turkish
        # This helps matching 'proje' with 'projesinde'
        tokens.append(t[:5])
    return tokens


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    if not union:
        return 0.0
    # Overlap weight: Intersection / Size of smaller set (to handle long contexts vs short questions better)
    # But for standard metrics we keep Jaccard
    return len(intersection) / len(union)


def evaluate_rag(
    question: str,
    chunks: List[Dict[str, Any]],
    answer: str,
    retrieval_confidence: float = 0.0,
    retrieval_quality: bool = False,
) -> Dict[str, Any]:
    # Clean answer: Extract only the 'Final Answer' part if CoT structure exists
    clean_answer = answer
    if "Final Answer:" in answer:
        parts = answer.split("Final Answer:", 1)
        clean_answer = parts[1].split("Sources Table:", 1)[0].strip()
    elif "Nihai Cevap:" in answer:
        parts = answer.split("Nihai Cevap:", 1)
        clean_answer = parts[1].split("Kaynak Tablosu:", 1)[0].strip()
    
    question_tokens = set(_tokenize(question))
    answer_tokens = set(_tokenize(clean_answer))

    context_sets: List[Set[str]] = []
    for chunk in chunks:
        context_sets.append(set(_tokenize(chunk.get("text", ""))))

    context_union: Set[str] = set()
    for tokens in context_sets:
        context_union |= tokens

    per_chunk_scores = [_jaccard(question_tokens, tokens) for tokens in context_sets if tokens]
    per_chunk_scores.sort(reverse=True)
    if per_chunk_scores:
        top_k = per_chunk_scores[: min(3, len(per_chunk_scores))]
        context_relevance = sum(top_k) / len(top_k)
    else:
        context_relevance = 0.0

    context_recall = (len(question_tokens & context_union) / len(question_tokens)) if question_tokens else 0.0
    faithfulness_jaccard = (len(answer_tokens & context_union) / len(answer_tokens)) if answer_tokens else 0.0
    answer_relevance_jaccard = (len(question_tokens & answer_tokens) / len(question_tokens)) if question_tokens else 0.0
 
    # Semantic Boost (Master Level)
    sem_faithfulness = _semantic_sim(clean_answer, " ".join([c.get("text", "") for c in chunks]))
    sem_answer_rel = _semantic_sim(question, clean_answer)
    
    # Hybrid Scores (Weighted Average)
    faithfulness = (faithfulness_jaccard * 0.3) + (sem_faithfulness * 0.7)
    answer_relevance = (answer_relevance_jaccard * 0.3) + (sem_answer_rel * 0.7)

    # Adjusted thresholds for professional/longer CoT-derived answers
    if context_relevance < 0.10 or context_recall < 0.10:
        failed_component = "retrieval"
        diagnosis = "Retriever alakasiz veya eksik baglam getiriyor."
    elif faithfulness < 0.20:
        failed_component = "generation"
        diagnosis = "Uretilen cevap baglama yeterince dayanmiyor (hallucination riski)."
    elif answer_relevance < 0.20:
        failed_component = "generation"
        diagnosis = "Cevap soruya yeterince dogrudan yanit vermiyor."
    else:
        failed_component = "none"
        diagnosis = "RAG cikti kalitesi metriklere gore kabul edilebilir."

    is_rel = (answer_relevance >= 0.20) and (context_relevance >= 0.08)
    is_sup = (faithfulness >= 0.20) and (retrieval_confidence >= 0.10 or retrieval_quality)
    is_use = (answer_relevance >= 0.20) and (len(clean_answer.strip()) >= 15)

    return {
        "context_relevance": round(context_relevance, 3),
        "context_recall": round(context_recall, 3),
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "confidence_score": round(retrieval_confidence, 3),
        "retrieval_quality": bool(retrieval_quality),
        "failed_component": failed_component,
        "diagnosis": diagnosis,
        "IsREL": bool(is_rel),
        "IsSUP": bool(is_sup),
        "IsUSE": bool(is_use),
        "question_token_count": len(question_tokens),
        "context_chunk_count": len(chunks),
    }

import uuid
from typing import List

from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    context_prefix: str = "",
) -> List[dict]:
    """
    Split cleaned text into overlapping chunks using recursive character splitting.
    If context_prefix is provided, it is prepended to every chunk to maintain global context.

    Args:
        text:           Cleaned text from clean_text().
        metadata:       Dict with at minimum: file_id, source_file, file_type.
        chunk_size:     Max characters per chunk (default: CHUNK_SIZE from config).
        chunk_overlap:  Characters shared with the next chunk (default: CHUNK_OVERLAP).
        context_prefix: Global document context to prepend to each chunk.

    Returns:
        List of chunk dicts.
    """
    if not text or not text.strip():
        return []

    raw_chunks = _split(text, chunk_size)
    overlapped = _apply_overlap(raw_chunks, chunk_overlap)
    total = len(overlapped)

    return [
        {
            "chunk_id": str(uuid.uuid4()),
            "file_id": metadata.get("file_id", ""),
            "source_file": metadata.get("source_file", ""),
            "file_type": metadata.get("file_type", ""),
            "username": metadata.get("username", ""),
            "chunk_index": idx,
            "total_chunks": total,
            "text": f"{context_prefix}\n\n{chunk}" if context_prefix else chunk,
            "char_count": len(chunk),
            "has_context": bool(context_prefix),
        }
        for idx, chunk in enumerate(overlapped)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split(text: str, chunk_size: int) -> List[str]:
    """
    Recursively split text into pieces that are each <= chunk_size chars.
    No overlap is applied here — overlap is added in a separate pass.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in _SEPARATORS:
        if sep == "":
            # Character-level fallback: hard-cut at chunk_size
            return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

        parts = text.split(sep)
        if len(parts) == 1:
            # This separator doesn't appear in text — try next
            continue

        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(_split(current, chunk_size))
                current = part

        if current:
            chunks.extend(_split(current, chunk_size))

        return [c for c in chunks if c.strip()]

    return [text]


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    Prepend the tail of the previous chunk to each subsequent chunk so that
    context is preserved across boundaries.
    Each chunk is produced exactly once — no duplication.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        result.append(tail + " " + chunks[i])

    return result
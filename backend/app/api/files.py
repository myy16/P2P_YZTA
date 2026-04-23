import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/files")
def list_files(username: Optional[str] = Query(default=None)):
    """
    Return the distinct files uploaded by the given user.
    Each entry: {file_id, original_name, chunk_count, size_mb}
    """
    try:
        store = get_vector_store()
        filters = {"username": {"$eq": username}} if username else None
        results = store.fetch_all(filters=filters)
    except Exception as exc:
        logger.exception("Failed to list files for user %s", username)
        raise HTTPException(status_code=500, detail="Could not retrieve file list.") from exc

    metadatas = results.get("metadatas") or []

    # Group by original_name — keep only the latest file_id per name
    by_name: dict = {}
    for meta in metadatas:
        fid = meta.get("file_id", "")
        name = meta.get("source_file", "")
        if not fid or not name:
            continue
        if name not in by_name:
            by_name[name] = {"file_id": fid, "original_name": name, "chunk_count": 0, "size_mb": 0.0}
        by_name[name]["chunk_count"] += 1

    return {"files": list(by_name.values())}

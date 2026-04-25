import logging
from functools import lru_cache
import hashlib
import math
from typing import List

logger = logging.getLogger(__name__)

from app.core.config import EMBEDDING_MODEL_NAME


class EmbeddingService:
    _fallback_dim = 384

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._use_fallback = False

    def _load_model(self):
        if self._use_fallback:
            return None
            
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
                import os as _os
                from app.core.config import HF_TOKEN
                
                # Force local files only to avoid hang/timeout on no-internet environments
                if HF_TOKEN:
                    _os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
                
                # Check if model exists in common cache locations or current dir
                # If we are in a restricted container, it's safer to just try and catch.
                # BUT: we must ensure we don't get a 'random uninitialized' model.
                # A trick is to check for modules.json or similar in the model path.
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
                logger.info("Successfully loaded embedding model: %s", self.model_name)
            except Exception as exc:
                logger.warning("Embedding model %s not found locally. Switching to deterministic fallback hash. (Error: %s)", self.model_name, exc)
                self._use_fallback = True
                self._model = None
        return self._model

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _fallback_embed(self, text: str) -> List[float]:
        vector = [0.0] * self._fallback_dim
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._fallback_dim
            weight = 1.0 + (len(token) / 10.0)
            vector[index] += weight

        return self._normalize(vector)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self._load_model()
        if model is not None:
            # Master Level: Batch size optimization
            batch_size = 32
            logger.info("Generating embeddings for %s texts (batch_size=%s)...", len(texts), batch_size)
            vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

        return [self._fallback_embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def pre_warm(self):
        """Warm up the model by loading it into memory."""
        self._load_model()

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
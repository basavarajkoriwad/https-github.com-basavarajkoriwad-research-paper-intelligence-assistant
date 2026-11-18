"""Model-singleton and disk-persisted SHA-256 embedding cache for CPU-efficient RAG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scholarmind.core.device import get_device
from scholarmind.core.logging import get_logger
from scholarmind.models.embeddings.retrieval_benchmark import (
    BaseTextEmbedder,
    SentenceTransformerEmbedder,
)

logger = get_logger(__name__)

# Process-wide model cache so SentenceTransformer weights load ONCE and are reused across requests/tests
_SHARED_EMBEDDERS: dict[tuple[str, str], SentenceTransformerEmbedder] = {}


def get_shared_sentence_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_cache_dir: Path | None = None,
    device: str = "auto",
) -> SentenceTransformerEmbedder:
    """Load a SentenceTransformer model once per process and reuse it."""
    resolved_device = str(get_device(device))
    key = (model_name, resolved_device)
    if key not in _SHARED_EMBEDDERS:
        cache_folder = None
        local_only = False
        if hf_cache_dir and (hf_cache_dir / "hub").exists():
            cache_folder = str(hf_cache_dir / "hub")
            local_only = True
        logger.info("Loading shared SentenceTransformer '%s' on %s...", model_name, resolved_device)
        _SHARED_EMBEDDERS[key] = SentenceTransformerEmbedder(
            model_name=model_name,
            device=resolved_device,
            cache_folder=cache_folder,
            local_files_only=local_only,
        )
    return _SHARED_EMBEDDERS[key]


class CachedTextEmbedder(BaseTextEmbedder):
    """Wraps any `BaseTextEmbedder` with a persistent SHA-256 embedding cache.

    Avoids recomputing embeddings for unchanged document chunks or repeated queries
    while preserving batch processing for cache misses.
    """

    def __init__(
        self,
        base_embedder: BaseTextEmbedder,
        cache_dir: Path | None = None,
    ) -> None:
        self.base_embedder = base_embedder
        self.name = base_embedder.name
        self.cache_dir = cache_dir
        self._cache: dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._keys_path = self.cache_dir / "embedding_cache_keys.json"
            self._vecs_path = self.cache_dir / "embedding_cache_vecs.npy"
            self._load_disk_cache()

    @staticmethod
    def _text_key(model_name: str, is_query: bool, text: str) -> str:
        prefix = "q:" if is_query else "d:"
        return hashlib.sha256(f"{model_name}|{prefix}{text}".encode("utf-8")).hexdigest()

    def _load_disk_cache(self) -> None:
        if self.cache_dir is None:
            return
        if self._keys_path.exists() and self._vecs_path.exists():
            try:
                keys: list[str] = json.loads(self._keys_path.read_text(encoding="utf-8"))
                vecs = np.load(self._vecs_path)
                if len(keys) == len(vecs):
                    self._cache = {k: vecs[i].astype(np.float32) for i, k in enumerate(keys)}
                    logger.info("Loaded %d cached embeddings from %s.", len(self._cache), self.cache_dir)
            except Exception as exc:
                logger.warning("Ignoring corrupt embedding cache (%s); rebuilding cleanly.", exc)
                self._cache = {}

    def _save_disk_cache(self) -> None:
        if self.cache_dir is None or not self._cache:
            return
        keys = list(self._cache.keys())
        vecs = np.vstack([self._cache[k] for k in keys]).astype(np.float32)
        self._keys_path.write_text(json.dumps(keys), encoding="utf-8")
        np.save(self._vecs_path, vecs)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [self._text_key(self.name, is_query, t) for t in texts]
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, (k, txt) in enumerate(zip(keys, texts)):
            if k in self._cache:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                miss_indices.append(idx)
                miss_texts.append(txt)

        if miss_texts:
            computed = self.base_embedder.encode(miss_texts, is_query=is_query)
            computed = np.asarray(computed, dtype=np.float32)
            for idx_pos, vec in zip(miss_indices, computed):
                self._cache[keys[idx_pos]] = vec
            self._save_disk_cache()

        return np.vstack([self._cache[k] for k in keys]).astype(np.float32)

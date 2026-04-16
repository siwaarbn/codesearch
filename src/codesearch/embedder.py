# handles embedding — turns code chunks into vectors using sentence-transformers
# caches results to disk so we don't redo work on files that haven't changed

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from codesearch.parser import CodeChunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"
_CACHE_FILE = "embed_cache.pkl"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | Path | None = None) -> None:
        # model download happens on first run (~80mb), cached locally after that
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path: Path | None = None
        if cache_dir is not None:
            self._cache_path = Path(cache_dir) / _CACHE_FILE
            if self._cache_path.exists():
                with open(self._cache_path, "rb") as f:
                    self._cache = pickle.load(f)

    def _save_cache(self) -> None:
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    @staticmethod
    def _key(text: str) -> str:
        # hash the text so the cache stays valid even if the file moves
        return hashlib.sha256(text.encode()).hexdigest()

    def embed_chunks(self, chunks: list[CodeChunk], batch_size: int = 64) -> np.ndarray:
        # not sure 64 is optimal here but seems fine in practice
        dim = self.model.get_sentence_embedding_dimension()
        results = np.zeros((len(chunks), dim), dtype=np.float32)

        uncached: list[tuple[int, str]] = []
        for i, chunk in enumerate(chunks):
            hit = self._cache.get(self._key(chunk.text))
            if hit is not None:
                results[i] = hit
            else:
                uncached.append((i, chunk.text))

        if uncached:
            idxs, texts = zip(*uncached)
            # normalize so dot product == cosine similarity in the index
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            for vec, (orig_idx, text) in zip(embeddings, uncached):
                results[orig_idx] = vec
                self._cache[self._key(text)] = vec
            self._save_cache()

        return results

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

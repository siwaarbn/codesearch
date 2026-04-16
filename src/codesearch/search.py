# takes a query embedding and finds the nearest neighbors in the index

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from codesearch.parser import CodeChunk


@dataclass
class SearchResult:
    name: str
    path: str
    language: str
    start_line: int
    end_line: int
    score: float
    preview: str  # first 3 lines of the chunk


def _preview(text: str, lines: int = 3) -> str:
    return "\n".join(text.splitlines()[:lines])


def search(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    chunks: list[CodeChunk],
    top_k: int = 5,
    lang_filter: str | None = None,
) -> list[SearchResult]:
    # fetch more candidates than we need so filtering doesn't leave us short
    fetch_k = min(index.ntotal, top_k * 4 if lang_filter else top_k)
    scores, indices = index.search(query_embedding.reshape(1, -1), fetch_k)

    results: list[SearchResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if lang_filter and chunk.language != lang_filter:
            continue
        results.append(SearchResult(
            name=chunk.name,
            path=chunk.path,
            language=chunk.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            score=float(score),
            preview=_preview(chunk.text),
        ))
        if len(results) >= top_k:
            break

    return results

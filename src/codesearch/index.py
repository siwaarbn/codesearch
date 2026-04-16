# builds and saves the faiss index
# IndexFlatIP + normalized vectors gives us cosine similarity
# metadata lives in a json file next to the binary index so it's easy to inspect

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from codesearch.parser import CodeChunk

_INDEX_FILE = "codesearch.index"
_META_FILE = "codesearch.meta.json"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, chunks: list[CodeChunk], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / _INDEX_FILE))
    # store everything as json — easier to debug than a binary format
    meta = {
        "chunks": [
            {
                "name": c.name,
                "path": c.path,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "text": c.text,
            }
            for c in chunks
        ]
    }
    (directory / _META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_index(directory: str | Path) -> tuple[faiss.IndexFlatIP, list[CodeChunk]]:
    directory = Path(directory)
    index = faiss.read_index(str(directory / _INDEX_FILE))
    # TODO: probably should validate this json before blindly deserializing it
    data = json.loads((directory / _META_FILE).read_text())
    chunks = [
        CodeChunk(
            name=c["name"],
            path=c["path"],
            language=c["language"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            text=c["text"],
        )
        for c in data["chunks"]
    ]
    return index, chunks


def indexed_paths(directory: str | Path) -> set[str]:
    """Return paths already in the index so we can skip them on re-index."""
    meta_file = Path(directory) / _META_FILE
    if not meta_file.exists():
        return set()
    data = json.loads(meta_file.read_text())
    return {c["path"] for c in data["chunks"]}


def add_to_index(
    new_embeddings: np.ndarray,
    new_chunks: list[CodeChunk],
    directory: str | Path,
) -> tuple[faiss.IndexFlatIP, list[CodeChunk]]:
    # load existing, tack on the new stuff, save back
    index, existing = load_index(directory)
    index.add(new_embeddings)
    merged = existing + new_chunks
    save_index(index, merged, directory)
    return index, merged

# fastapi app — handles zip uploads, indexing, and search
# sessions are just an in-memory dict, resets on restart which is fine for now

from __future__ import annotations

import io
import shutil
import tempfile
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from codesearch.embedder import Embedder
from codesearch.index import build_index, load_index, save_index
from codesearch.parser import parse_directory
from codesearch.search import search as _search

# -------------------------------------------------------------------
# paths
# -------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent        # absolute path to src/codesearch/
_REPO_ROOT = _HERE.parents[1]                 # src/codesearch -> src -> project root
WEB_DIR = _REPO_ROOT / "web"

# -------------------------------------------------------------------
# session store
# -------------------------------------------------------------------

_sessions: OrderedDict[str, Path] = OrderedDict()
_DEMO_ID = "demo"
_MAX_SESSIONS = 50  # evict oldest when we go over this
_SAFE_EXTS = frozenset({".py", ".js", ".jsx", ".mjs"})
_MAX_ZIP_BYTES = 50 * 1024 * 1024  # 50mb should be plenty


@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    # load once, reuse everywhere — model is ~80mb so we don't want to reload it
    return Embedder()


def _register_session(sid: str, path: Path) -> None:
    _sessions[sid] = path
    non_demo = [k for k in list(_sessions) if k != _DEMO_ID]
    while len(non_demo) > _MAX_SESSIONS:
        oldest = non_demo.pop(0)
        shutil.rmtree(_sessions.pop(oldest), ignore_errors=True)


def _build_demo() -> None:
    # index the codesearch source itself so visitors can try without uploading
    # _HERE is the resolved absolute path to this file's directory, so this
    # works regardless of the cwd the server was launched from
    src = _HERE  # src/codesearch/ — the package directory itself
    chunks = parse_directory(src, languages=["python"])
    if not chunks:
        raise RuntimeError("no python files found in package source")
    for c in chunks:
        try:
            c.path = str(Path(c.path).relative_to(_HERE.parent))
        except ValueError:
            pass
    embs = _embedder().embed_chunks(chunks)
    d = Path(tempfile.mkdtemp(prefix="cs_demo_"))
    save_index(build_index(embs), chunks, d)
    _sessions[_DEMO_ID] = d
    print(f"demo index built: {len(chunks)} chunks")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # try to build demo eagerly so it's ready on first request
    # if it fails (e.g. model not cached yet) it will be retried lazily
    try:
        _build_demo()
    except Exception as e:
        print(f"warning: demo build failed at startup, will retry on first request: {e}")
    yield
    for p in _sessions.values():
        shutil.rmtree(p, ignore_errors=True)


# -------------------------------------------------------------------
# app
# -------------------------------------------------------------------

app = FastAPI(title="codesearch", lifespan=_lifespan)


# -------------------------------------------------------------------
# schemas
# -------------------------------------------------------------------

class SearchRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5


class ResultItem(BaseModel):
    name: str
    path: str
    language: str
    start_line: int
    score: float
    preview: str


class SearchResponse(BaseModel):
    results: list[ResultItem]


class SessionInfo(BaseModel):
    session_id: str
    num_chunks: int


# -------------------------------------------------------------------
# endpoints
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    if not (WEB_DIR / "index.html").exists():
        raise HTTPException(404, "frontend not found — did you forget to copy the web/ directory?")
    return FileResponse(WEB_DIR / "index.html")


@app.get("/demo", response_model=SessionInfo)
def demo():
    if _DEMO_ID not in _sessions:
        # startup build failed — try now so the user isn't permanently locked out
        try:
            _build_demo()
        except Exception as e:
            raise HTTPException(503, f"demo unavailable: {e}")
    _, chunks = load_index(_sessions[_DEMO_ID])
    return SessionInfo(session_id=_DEMO_ID, num_chunks=len(chunks))


@app.post("/index", response_model=SessionInfo)
async def index_zip(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".zip"):
        raise HTTPException(400, "please upload a .zip file")

    data = await file.read()
    if len(data) > _MAX_ZIP_BYTES:
        raise HTTPException(413, "zip is too large (50mb max)")

    extract = Path(tempfile.mkdtemp(prefix="cs_ext_"))
    try:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    p = Path(name)
                    # zip slip check — skip anything sketchy
                    if p.is_absolute() or ".." in p.parts:
                        continue
                    if p.suffix.lower() not in _SAFE_EXTS:
                        continue
                    dest = extract / p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
        except zipfile.BadZipFile:
            raise HTTPException(400, "couldn't open that zip file")

        chunks = parse_directory(extract)
        if not chunks:
            raise HTTPException(400, "no .py or .js files found in the zip")

        # store relative paths so they display nicely
        for c in chunks:
            try:
                c.path = str(Path(c.path).relative_to(extract))
            except ValueError:
                pass

        embs = _embedder().embed_chunks(chunks, batch_size=32)
        index_dir = Path(tempfile.mkdtemp(prefix="cs_idx_"))
        save_index(build_index(embs), chunks, index_dir)
    finally:
        shutil.rmtree(extract, ignore_errors=True)

    sid = str(uuid.uuid4())
    _register_session(sid, index_dir)
    return SessionInfo(session_id=sid, num_chunks=len(chunks))


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    d = _sessions.get(req.session_id)
    if d is None:
        raise HTTPException(404, "session not found — please re-upload your code")

    faiss_idx, chunks = load_index(d)
    hits = _search(
        _embedder().embed_query(req.query),
        faiss_idx,
        chunks,
        top_k=min(req.top_k, 20),
    )
    return SearchResponse(results=[
        ResultItem(
            name=h.name, path=h.path, language=h.language,
            start_line=h.start_line, score=h.score, preview=h.preview,
        )
        for h in hits
    ])

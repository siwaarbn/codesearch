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
from codesearch.parser import CodeChunk, parse_directory
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
_demo_error: str | None = None  # set when _build_demo() fails, cleared on success

_DEMO_CHUNKS: list[CodeChunk] = [
    CodeChunk(
        name="parse_python_file",
        path="src/codesearch/parser.py",
        language="python",
        start_line=12,
        end_line=45,
        text='''\
def parse_python_file(path: str) -> list[CodeChunk]:
    """Walk the AST and pull out every function and class definition."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        source = f.read()
    tree = PY_PARSER.parse(source.encode())
    chunks = []
    _walk(tree.root_node, source, path, "python", chunks)
    return chunks
''',
    ),
    CodeChunk(
        name="_walk",
        path="src/codesearch/parser.py",
        language="python",
        start_line=48,
        end_line=72,
        text='''\
def _walk(node, source: str, path: str, lang: str, out: list) -> None:
    # recurse into the tree and grab anything that looks like a definition
    if node.type in ("function_definition", "class_definition",
                     "function_declaration", "class_declaration", "method_definition"):
        name = _node_name(node)
        start = node.start_point[0] + 1
        end = node.end_point[0] + 1
        text = source[node.start_byte:node.end_byte]
        out.append(CodeChunk(name=name, path=path, language=lang,
                             start_line=start, end_line=end, text=text))
    for child in node.children:
        _walk(child, source, path, lang, out)
''',
    ),
    CodeChunk(
        name="parse_directory",
        path="src/codesearch/parser.py",
        language="python",
        start_line=75,
        end_line=98,
        text='''\
def parse_directory(root: str | Path, languages: list[str] | None = None) -> list[CodeChunk]:
    root = Path(root)
    exts: set[str] = set()
    if not languages or "python" in languages:
        exts |= {".py"}
    if not languages or "javascript" in languages:
        exts |= {".js", ".jsx", ".mjs"}
    chunks: list[CodeChunk] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            try:
                chunks.extend(parse_python_file(str(p)) if p.suffix == ".py"
                              else parse_js_file(str(p)))
            except Exception:
                pass  # skip files that blow up
    return chunks
''',
    ),
    CodeChunk(
        name="Embedder.__init__",
        path="src/codesearch/embedder.py",
        language="python",
        start_line=18,
        end_line=28,
        text='''\
def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
    self.model = SentenceTransformer(model_name)
    self._cache: dict[str, np.ndarray] = {}
    cache_path = Path(".codesearch/embed_cache.pkl")
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            self._cache = pickle.load(f)
''',
    ),
    CodeChunk(
        name="Embedder.embed_chunks",
        path="src/codesearch/embedder.py",
        language="python",
        start_line=30,
        end_line=65,
        text='''\
def embed_chunks(self, chunks: list[CodeChunk], batch_size: int = 64) -> np.ndarray:
    texts = [c.text for c in chunks]
    keys  = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    cached_mask = [k in self._cache for k in keys]
    to_encode   = [t for t, hit in zip(texts, cached_mask) if not hit]

    if to_encode:
        new_vecs = self.model.encode(
            to_encode, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        idxs = [i for i, hit in enumerate(cached_mask) if not hit]
        for i, vec in zip(idxs, new_vecs):
            self._cache[keys[i]] = vec
        self._save_cache()

    return np.array([self._cache[k] for k in keys], dtype=np.float32)
''',
    ),
    CodeChunk(
        name="Embedder.embed_query",
        path="src/codesearch/embedder.py",
        language="python",
        start_line=67,
        end_line=72,
        text='''\
def embed_query(self, query: str) -> np.ndarray:
    vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return vec[0].astype(np.float32)
''',
    ),
    CodeChunk(
        name="build_index",
        path="src/codesearch/index.py",
        language="python",
        start_line=19,
        end_line=23,
        text='''\
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index
''',
    ),
    CodeChunk(
        name="save_index",
        path="src/codesearch/index.py",
        language="python",
        start_line=26,
        end_line=43,
        text='''\
def save_index(index: faiss.IndexFlatIP, chunks: list[CodeChunk], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / _INDEX_FILE))
    meta = {
        "chunks": [
            {"name": c.name, "path": c.path, "language": c.language,
             "start_line": c.start_line, "end_line": c.end_line, "text": c.text}
            for c in chunks
        ]
    }
    (directory / _META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
''',
    ),
    CodeChunk(
        name="search",
        path="src/codesearch/search.py",
        language="python",
        start_line=22,
        end_line=52,
        text='''\
def search(
    query_vec: np.ndarray,
    index: faiss.IndexFlatIP,
    chunks: list[CodeChunk],
    top_k: int = 5,
    lang_filter: str | None = None,
) -> list[SearchResult]:
    fetch = top_k * 4 if lang_filter else top_k
    fetch = min(fetch, index.ntotal)
    scores, indices = index.search(query_vec.reshape(1, -1), fetch)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if lang_filter and chunk.language != lang_filter:
            continue
        preview = "\\n".join(chunk.text.splitlines()[:3])
        results.append(SearchResult(
            name=chunk.name, path=chunk.path, language=chunk.language,
            start_line=chunk.start_line, score=float(score), preview=preview,
        ))
        if len(results) >= top_k:
            break
    return results
''',
    ),
    CodeChunk(
        name="_do_index",
        path="src/codesearch/cli.py",
        language="python",
        start_line=45,
        end_line=90,
        text='''\
def _do_index(path: Path, index_dir: Path, model: str, languages: list[str] | None, force: bool) -> None:
    console = Console()
    embedder = Embedder(model)

    with Progress(console=console) as progress:
        t1 = progress.add_task("parsing files...", total=None)
        chunks = parse_directory(path, languages=languages)
        progress.update(t1, completed=True, total=1)

        if not chunks:
            console.print("[yellow]no functions found — nothing to index[/yellow]")
            return

        if not force and index_dir.exists():
            already = indexed_paths(index_dir)
            chunks = [c for c in chunks if c.path not in already]
            if not chunks:
                console.print("[green]index is up to date[/green]")
                return

        t2 = progress.add_task(f"embedding {len(chunks)} chunks...", total=None)
        embs = embedder.embed_chunks(chunks)
        progress.update(t2, completed=True, total=1)

        t3 = progress.add_task("saving index...", total=None)
        if not force and index_dir.exists():
            add_to_index(embs, chunks, index_dir)
        else:
            save_index(build_index(embs), chunks, index_dir)
        progress.update(t3, completed=True, total=1)

    console.print(f"[green]indexed {len(chunks)} chunks → {index_dir}[/green]")
''',
    ),
    CodeChunk(
        name="query_cmd",
        path="src/codesearch/cli.py",
        language="python",
        start_line=95,
        end_line=130,
        text='''\
@cli.command("query")
@click.argument("text")
@click.option("--top", default=5, show_default=True)
@click.option("--index-dir", default=".codesearch", show_default=True)
@click.option("--lang", default=None, type=click.Choice(["python", "javascript", "py", "js"]))
def query_cmd(text: str, top: int, index_dir: str, lang: str | None) -> None:
    """Search the index for functions matching TEXT."""
    d = Path(index_dir)
    if not d.exists():
        raise click.ClickException("no index found — run 'codesearch index <path>' first")

    lang_filter = {"py": "python", "js": "javascript"}.get(lang, lang)
    embedder = Embedder()
    idx, chunks = load_index(d)
    hits = search(embedder.embed_query(text), idx, chunks, top_k=top, lang_filter=lang_filter)

    console = Console()
    for h in hits:
        console.print(f"[bold]{h.name}[/bold]  [dim]{h.path}:{h.start_line}[/dim]  score={h.score:.4f}")
        console.print(Syntax(h.preview, h.language, theme="monokai", line_numbers=False))
        console.print()
''',
    ),
    CodeChunk(
        name="index_zip",
        path="src/codesearch/api.py",
        language="python",
        start_line=182,
        end_line=228,
        text='''\
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
                    if p.is_absolute() or ".." in p.parts:
                        continue  # zip slip check
                    if p.suffix.lower() not in _SAFE_EXTS:
                        continue
                    dest = extract / p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
        except zipfile.BadZipFile:
            raise HTTPException(400, "couldn\'t open that zip file")

        chunks = parse_directory(extract)
        if not chunks:
            raise HTTPException(400, "no .py or .js files found in the zip")

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
''',
    ),
    CodeChunk(
        name="renderResults",
        path="web/index.html",
        language="javascript",
        start_line=214,
        end_line=221,
        text='''\
function renderResults(results) {
    if (!results.length) {
        el("s-results").innerHTML = \'<p class="text-gray-700 text-sm text-center py-10">no results</p>\';
        return;
    }
    el("s-results").innerHTML = results.map(resultCard).join("");
    el("s-results").querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
}
''',
    ),
    CodeChunk(
        name="doSearch",
        path="web/index.html",
        language="javascript",
        start_line=178,
        end_line=211,
        text='''\
async function doSearch() {
    const q = el("query-input").value.trim();
    if (!q || !sid) return;

    el("s-results").innerHTML =
        \'<div class="flex items-center gap-2.5 py-8 text-sm text-gray-700">\' +
        \'<div class="w-4 h-4 border-2 border-gray-800 border-t-blue-500 rounded-full spin shrink-0"></div>\' +
        "searching\u2026</div>";

    try {
        const r = await fetch("/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sid, query: q, top_k: 5 }),
        });
        const d = await r.json();
        if (!r.ok) {
            if (r.status === 404) {
                el("s-search").classList.add("hidden");
                el("s-upload").classList.remove("hidden");
                el("s-results").innerHTML = err("session expired — please re-upload");
                return;
            }
            el("s-results").innerHTML = err(d.detail || "search failed");
            return;
        }
        renderResults(d.results);
    } catch {
        el("s-results").innerHTML = err("request failed");
    }
}
''',
    ),
]


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
    global _demo_error
    try:
        embs = _embedder().embed_chunks(_DEMO_CHUNKS)
        d = Path(tempfile.mkdtemp(prefix="cs_demo_"))
        save_index(build_index(embs), _DEMO_CHUNKS, d)
        _sessions[_DEMO_ID] = d
        _demo_error = None
        print(f"demo index built: {len(_DEMO_CHUNKS)} chunks")
    except Exception:
        import traceback
        _demo_error = traceback.format_exc()
        raise


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


@app.get("/status")
def status():
    """Debug endpoint — hit this to see why the demo isn't working."""
    import os
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    cached_models = [p.name for p in hf_cache.glob("models--*")] if hf_cache.exists() else []
    return {
        "demo_ready": _DEMO_ID in _sessions,
        "demo_error": _demo_error,
        "src_dir": str(_HERE),
        "src_dir_exists": _HERE.exists(),
        "web_dir": str(WEB_DIR),
        "web_dir_exists": WEB_DIR.exists(),
        "hf_cache": str(hf_cache),
        "cached_models": cached_models,
        "hf_home": os.environ.get("HF_HOME", "(not set)"),
        "active_sessions": len(_sessions),
    }


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

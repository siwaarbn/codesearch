# CLI — two commands: `codesearch index` and `codesearch query`
# uses rich for nicer output

from __future__ import annotations

import threading
import time
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.text import Text

from codesearch.embedder import Embedder
from codesearch.index import add_to_index, build_index, indexed_paths, load_index, save_index
from codesearch.parser import parse_directory
from codesearch.search import search

_DEFAULT_INDEX_DIR = ".codesearch"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_WATCHABLE_EXTS = frozenset({".py", ".js", ".jsx", ".mjs"})

# accept "py" and "js" as shorthands
_LANG_ALIASES = {"py": "python", "js": "javascript", "python": "python", "javascript": "javascript"}

console = Console()


def _resolve_lang(lang: str | None) -> str | None:
    return _LANG_ALIASES.get(lang.lower(), lang.lower()) if lang else None


def _do_index(path: str, index_dir: str, model: str, lang: str | None, force: bool = False) -> None:
    languages = [lang] if lang else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}[/]/[cyan]{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t1 = progress.add_task("[green]Parsing...", total=None)
        all_chunks = parse_directory(path, languages=languages)
        progress.update(t1, total=1, completed=1)

        if not all_chunks:
            console.print("[yellow]No code chunks found.[/]")
            return

        already = set() if force else indexed_paths(index_dir)
        new_chunks = [c for c in all_chunks if c.path not in already]

        if not new_chunks:
            console.print(
                f"[green]✓[/] Index up to date — "
                f"[cyan]{len(all_chunks)}[/] chunks already indexed."
            )
            return

        t2 = progress.add_task("[blue]Embedding...", total=len(new_chunks))
        embedder = Embedder(model, cache_dir=index_dir)
        batch_size = 64
        batches: list[np.ndarray] = []
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            batches.append(embedder.embed_chunks(batch, batch_size=batch_size))
            progress.advance(t2, len(batch))
        embeddings = np.vstack(batches)

        t3 = progress.add_task("[magenta]Saving...", total=1)
        if already and not force:
            add_to_index(embeddings, new_chunks, index_dir)
        else:
            save_index(build_index(embeddings), new_chunks, index_dir)
        progress.advance(t3)

    console.print(
        f"[bold green]✓[/] Indexed [cyan]{len(new_chunks)}[/] new chunks "
        f"([cyan]{len(all_chunks)}[/] total) → [dim]{index_dir}/[/]"
    )


@click.group()
def cli() -> None:
    """codesearch — find code by meaning, not keywords"""


@cli.command("index")
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--index-dir", default=_DEFAULT_INDEX_DIR, show_default=True)
@click.option("--model", default=_DEFAULT_MODEL, show_default=True)
@click.option("--lang", default=None, help="python or javascript (default: both)")
@click.option("--watch", is_flag=True, help="re-index when files change")
def index_cmd(path: str, index_dir: str, model: str, lang: str | None, watch: bool) -> None:
    """Walk PATH, parse source files, build a search index."""
    lang = _resolve_lang(lang)
    _do_index(path, index_dir, model, lang)

    if not watch:
        return

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        console.print("[red]watchdog not installed. Run: uv add watchdog[/]")
        raise SystemExit(1)

    debounce: threading.Timer | None = None

    class _Handler(FileSystemEventHandler):
        def _trigger(self, event_path: str) -> None:
            if Path(event_path).suffix not in _WATCHABLE_EXTS:
                return
            nonlocal debounce
            if debounce:
                debounce.cancel()
            # TODO: this re-indexes everything on each change which is wasteful
            # would be better to only re-process the modified file
            debounce = threading.Timer(
                1.0, _do_index, args=(path, index_dir, model, lang), kwargs={"force": True}
            )
            debounce.start()

        def on_modified(self, event):
            if not event.is_directory:
                self._trigger(event.src_path)

        on_created = on_modified

    observer = Observer()
    observer.schedule(_Handler(), path=path, recursive=True)
    observer.start()
    console.print(f"[dim]Watching [cyan]{path}[/] for changes. Ctrl+C to stop.[/]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


@cli.command("query")
@click.argument("text")
@click.option("--index-dir", default=_DEFAULT_INDEX_DIR, show_default=True)
@click.option("--model", default=_DEFAULT_MODEL, show_default=True)
@click.option("--top", default=5, show_default=True, help="number of results")
@click.option("--lang", default=None, help="filter by language")
def query_cmd(text: str, index_dir: str, model: str, top: int, lang: str | None) -> None:
    """Search the index for TEXT."""
    lang = _resolve_lang(lang)
    index_path = Path(index_dir)

    if not (index_path / "codesearch.index").exists():
        console.print(
            f"[red]No index found at '{index_dir}'. Run [bold]codesearch index[/] first.[/]"
        )
        raise SystemExit(1)

    faiss_index, chunks = load_index(index_path)
    embedder = Embedder(model)
    query_vec = embedder.embed_query(text)
    results = search(query_vec, faiss_index, chunks, top_k=top, lang_filter=lang)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    console.print(f'\n[bold]Results for[/] [bold cyan]"{text}"[/]\n')

    lang_colors = {"python": "green", "javascript": "yellow"}

    for rank, result in enumerate(results, 1):
        lcolor = lang_colors.get(result.language, "white")
        title = Text.assemble(
            (f" {rank} ", "bold white on blue"),
            "  ",
            (result.name, "bold white"),
            "  ",
            (f"[{result.language}]", lcolor),
            "  ",
            (f"{result.path}:{result.start_line}", "dim"),
        )
        syntax = Syntax(
            result.preview,
            result.language,
            theme="monokai",
            line_numbers=True,
            start_line=result.start_line,
        )
        console.print(Panel(
            syntax,
            title=title,
            subtitle=f"[dim]score: {result.score:.4f}[/]",
            border_style="bright_black",
        ))

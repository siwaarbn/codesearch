# parses python and javascript source files and pulls out every function/class
# uses tree-sitter which is way better than trying to do this with regex

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# TODO: add TypeScript support at some point
_LANG_CONFIGS: dict[str, dict] = {
    "python": {
        "language": Language(tspython.language()),
        "extensions": frozenset({".py"}),
        "chunk_types": frozenset({"function_definition", "class_definition"}),
    },
    "javascript": {
        "language": Language(tsjs.language()),
        "extensions": frozenset({".js", ".jsx", ".mjs"}),
        "chunk_types": frozenset({
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
        }),
    },
}

_EXT_TO_LANG: dict[str, str] = {
    ext: lang
    for lang, cfg in _LANG_CONFIGS.items()
    for ext in cfg["extensions"]
}


@dataclass
class CodeChunk:
    name: str
    path: str
    language: str
    start_line: int
    end_line: int
    text: str


def _node_name(node, source: bytes) -> str:
    # tree-sitter gives us a "name" field on most definitions
    # arrow functions etc. won't have one so we just call them <anonymous>
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
    return "<anonymous>"


def _extract_chunks(tree, source: bytes, path: str, language: str, chunk_types: frozenset) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    def walk(node) -> None:
        if node.type in chunk_types:
            chunks.append(CodeChunk(
                name=_node_name(node, source),
                path=path,
                language=language,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                text=source[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
            ))
        # keep recursing even after a match so we catch nested defs too
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def parse_file(path: str | Path) -> list[CodeChunk]:
    path = Path(path)
    lang_name = _EXT_TO_LANG.get(path.suffix.lower())
    if not lang_name:
        return []
    cfg = _LANG_CONFIGS[lang_name]
    source = path.read_bytes()
    parser = Parser(cfg["language"])
    tree = parser.parse(source)
    return _extract_chunks(tree, source, str(path), lang_name, cfg["chunk_types"])


def parse_directory(root: str | Path, languages: list[str] | None = None) -> list[CodeChunk]:
    root = Path(root)
    if languages:
        allowed_exts = frozenset(
            ext
            for lang in languages
            for ext in _LANG_CONFIGS.get(lang, {}).get("extensions", frozenset())
        )
    else:
        allowed_exts = frozenset(_EXT_TO_LANG)

    chunks: list[CodeChunk] = []
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in allowed_exts:
            chunks.extend(parse_file(file))
    return chunks

from codesearch.cli import cli as main

__all__ = ["main", "serve"]


def serve() -> None:
    import uvicorn
    uvicorn.run("codesearch.api:app", host="0.0.0.0", port=8000, reload=False)

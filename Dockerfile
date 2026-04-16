FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# install deps first so this layer gets cached between code-only changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# bake the embedding model into the image so the first request isn't painfully slow
# this adds ~120mb to the image but avoids a 30s download on cold start
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ src/
COPY web/ web/

EXPOSE 8000

# $PORT is injected by Railway; fall back to 8000 locally
CMD ["sh", "-c", "uv run uvicorn codesearch.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM python:3.11-slim

WORKDIR /app

# set a fixed cache location so the model downloaded at build time
# is found in the same place at runtime
ENV HF_HOME=/app/.cache/huggingface

RUN pip install --no-cache-dir uv

# install deps first so this layer gets cached between code-only changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# download and cache the embedding model during the build
# so the app starts instantly and has no runtime dependency on huggingface.co
RUN uv run python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
# sanity check — embed one string to confirm the model actually works
v = m.encode(['hello world'])
assert v.shape == (1, 384), f'unexpected shape: {v.shape}'
print('model ok, embedding dim:', v.shape[1])
"

COPY src/ src/
COPY web/ web/

EXPOSE 8000

# $PORT is injected by Railway; fall back to 8000 locally
CMD ["sh", "-c", "uv run uvicorn codesearch.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

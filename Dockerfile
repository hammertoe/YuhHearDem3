FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG VITE_COMMIT_SHA=dev
ENV VITE_COMMIT_SHA=$VITE_COMMIT_SHA

RUN cd frontend && npm ci && npm run build && cp public/parliament-bg.jpg dist/

# Pre-download the BGE embedding model so the runtime container doesn't need
# network access at first boot to populate the sentence-transformers cache.
ARG LOCAL_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
ENV LOCAL_EMBEDDING_MODEL=${LOCAL_EMBEDDING_MODEL}
RUN mkdir -p /app/.cache/sentence-transformers \
    && python -c "import os; from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer(os.environ['LOCAL_EMBEDDING_MODEL'], cache_folder='/app/.cache/sentence-transformers'); \
print('baked', m.model_card_data.base_model if m.model_card_data else 'model')"

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell sentence-transformers where to find the pre-baked BGE model.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "api.search_api:app", "--host", "0.0.0.0", "--port", "8000"]

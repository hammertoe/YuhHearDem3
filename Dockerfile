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

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "api.search_api:app", "--host", "0.0.0.0", "--port", "8000"]

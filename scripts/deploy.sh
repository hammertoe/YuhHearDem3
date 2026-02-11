#!/bin/bash
set -e

# Production Deployment Script for YuhHearDem
# Usage: ./scripts/deploy.sh

APP_DIR="/opt/yuhheardem"
DOMAIN="beta.yuhheardem.com"
PORT=8003

echo "🚀 Starting YuhHearDem deployment..."

# Pull latest code
cd "$APP_DIR"
git fetch
git reset --hard origin/main

# Build frontend
cd "$APP_DIR/frontend"
npm ci
npm run build

# Build Docker image
cd "$APP_DIR"
docker build -t yuhheardem:latest .

# Stop old container
docker rm -f yhd-web 2>/dev/null || true

# Run new container
docker run -d \
    --name yhd-web \
    --network host \
    -e DATABASE_URL="${DATABASE_URL}" \
    -e GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
    -e GOOGLE_API_KEY_2="${GOOGLE_API_KEY_2}" \
    -e GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}" \
    -e MEMGRAPH_HOST="${MEMGRAPH_HOST:-localhost}" \
    -e MEMGRAPH_PORT="${MEMGRAPH_PORT:-7687}" \
    --restart unless-stopped \
    yuhheardem:latest

# Wait for container to be healthy
echo "⏳ Waiting for container to start..."
sleep 5

# Test health endpoint
for i in {1..10}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Container is healthy!"
        break
    fi
    echo "⏳ Waiting for health check... ($i/10)"
    sleep 2
done

echo "✅ Deployment complete at http://$DOMAIN"

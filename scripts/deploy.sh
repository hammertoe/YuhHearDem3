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
echo "⏳ Waiting for container to start..."
sleep 5

# Test health endpoint
health_passed=false
for i in {1..10}; do
  health_response=$(curl -s -w "\n%{http_code}" "http://localhost:$PORT/health" 2>&1)
  health_code=$(echo "$health_response" | tail -1)
  if [ "$health_code" = "200" ]; then
    health_passed=true
    echo "✅ Container is healthy! (HTTP $health_code)"
    break
  else
    echo "⏳ Health check failed (HTTP $health_code)... ($i/10)"
  fi
  sleep 2
done

if [ "$health_passed" = false ]; then
  echo "❌ Health check failed after 10 attempts. Continuing anyway..."
fi

echo "✅ Deployment complete at http://$DOMAIN"

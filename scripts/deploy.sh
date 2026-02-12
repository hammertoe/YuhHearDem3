#!/bin/bash
set -e

# Production Deployment Script for YuhHearDem
# Usage: ./scripts/deploy.sh

APP_DIR="/opt/yuhheardem"
DOMAIN="beta.yuhheardem.com"
PORT=8000
ENV_FILE="$APP_DIR/.env"
STATIC_DIR="/var/lib/yuhheardem/static"

echo "🚀 Starting YuhHearDem deployment..."

# Pull latest code
cd "$APP_DIR"
git fetch
git reset --hard origin/main

# Build frontend
cd "$APP_DIR/frontend"
npm ci
npm run build

# Sync frontend static files for nginx
mkdir -p "$STATIC_DIR"
rsync -a --delete "$APP_DIR/frontend/dist/" "$STATIC_DIR/"
find "$STATIC_DIR" -name '._*' -delete
chown -R www-data:www-data "$STATIC_DIR"

# Build Docker image
cd "$APP_DIR"
COMMIT_SHA=$(git rev-parse --short HEAD)
docker build --build-arg VITE_COMMIT_SHA="$COMMIT_SHA" -t yuhheardem:latest .

# Stop old container
docker rm -f yhd-web 2>/dev/null || true

# Run new container
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Missing env file: $ENV_FILE"
    exit 1
fi

docker run -d \
    --name yhd-web \
    --network host \
    --env-file "$ENV_FILE" \
    -e GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}" \
    --restart unless-stopped \
    yuhheardem:latest

# Wait for container to be healthy
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

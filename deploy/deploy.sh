#!/bin/bash
# deploy.sh
# Blue-green deployment for YuhHearDem3
# Run as: matt (needs sudo for nginx reload)
#
# Usage:
#   ./deploy.sh <version>  # Deploy specific version (e.g., sha, staging, latest)
#   ./deploy.sh rollback   # Switch back to previous slot
#   ./deploy.sh status     # Show current deployment status
#
# Prerequisites:
#   - Docker logged in to ghcr.io
#   - GHCR_TOKEN environment variable or ~/.ghcr-token file

set -euo pipefail

# Configuration
APP_NAME="yhd"
DEPLOY_DIR="/opt/yuhheardem3"
STATE_FILE="/var/lib/yuhheardem3/active-slot"
VERSION_FILE="/var/lib/yuhheardem3/deployed-version"
UPSTREAM_CONF="/etc/nginx/conf.d/yhd-upstream.conf"

# Container registry
REGISTRY="ghcr.io"
IMAGE_OWNER="hammertoe"
IMAGE_PREFIX="${REGISTRY}/${IMAGE_OWNER}/yhd"

# Health check settings
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Slot management
get_active_slot() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo "blue"
    fi
}

get_inactive_slot() {
    local active
    active=$(get_active_slot)
    if [ "$active" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

get_slot_port() {
    if [ "$1" = "blue" ]; then
        echo "8003"
    else
        echo "8013"
    fi
}

get_compose_file() {
    echo "${DEPLOY_DIR}/docker-compose.${1}.yml"
}

get_deployed_version() {
    if [ -f "$VERSION_FILE" ]; then
        cat "$VERSION_FILE"
    else
        echo "unknown"
    fi
}

# Clean up old Docker images
cleanup_docker_images() {
    log_info "Cleaning up old Docker images..."
    sudo docker image prune -af --filter "until=168h" >/dev/null 2>&1 || true
    log_success "Docker cleanup complete"
}

# Health check for a slot
health_check_slot() {
    local slot=$1
    local port
    port=$(get_slot_port "$slot")
    local url="http://127.0.0.1:${port}/health"

    log_info "Checking health at ${url}..."

    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        local response
        response=$(curl -sf "${url}" 2>/dev/null)
        local curl_status=$?

        if [ $curl_status -eq 0 ]; then
            local status
            status=$(echo "$response" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)

            if [ "$status" = "healthy" ] || [ "$status" = "degraded" ] || [ "$status" = "ok" ]; then
                log_success "Health check passed on attempt ${i} (status: ${status})"
                return 0
            else
                log_info "Health check attempt ${i}/${HEALTH_CHECK_RETRIES}: status is '${status}', retrying..."
            fi
        else
            log_info "Health check attempt ${i}/${HEALTH_CHECK_RETRIES}: no response, retrying..."
        fi
        sleep $HEALTH_CHECK_INTERVAL
    done

    log_error "Health check failed after ${HEALTH_CHECK_RETRIES} attempts"
    curl -s "${url}" 2>/dev/null | head -c 500 || true
    return 1
}

# Switch nginx upstream to new slot
switch_upstream() {
    local new_slot=$1
    local new_port
    new_port=$(get_slot_port "$new_slot")

    log_info "Switching nginx upstream to ${new_slot} (port ${new_port})..."

    sudo tee "$UPSTREAM_CONF" > /dev/null <<EOF
# Active upstream for YuhHearDem3
# Managed by deploy.sh - do not edit manually
# Switched to ${new_slot} at $(date -Iseconds)
upstream yhd_backend {
    server 127.0.0.1:${new_port};
}
EOF

    sudo nginx -t
    sudo systemctl reload nginx

    echo "$new_slot" | sudo tee "$STATE_FILE" > /dev/null

    log_success "Nginx switched to ${new_slot}"
}

# Ensure logged in to GitHub Container Registry
ensure_ghcr_login() {
    log_info "Checking GitHub Container Registry login..."

    if ! docker pull "${IMAGE_PREFIX}-gateway:latest" >/dev/null 2>&1; then
        log_warning "Not logged in to ghcr.io, attempting login..."

        local token="${GHCR_TOKEN:-}"
        if [ -z "$token" ] && [ -f "$HOME/.ghcr-token" ]; then
            token=$(cat "$HOME/.ghcr-token")
        fi

        if [ -z "$token" ]; then
            log_error "No GHCR_TOKEN found. Set GHCR_TOKEN env var or create ~/.ghcr-token file"
            log_error "Get a token from: https://github.com/settings/tokens (need read:packages scope)"
            exit 1
        fi

        echo "$token" | docker login ghcr.io -u "${GHCR_USER:-${IMAGE_OWNER}}" --password-stdin
        log_success "Logged in to ghcr.io"
    else
        log_success "Already logged in to ghcr.io"
    fi
}

# Pull images for a specific version
pull_images() {
    local version=$1
    local slot=$2

    log_info "Pulling images for version ${version}..."

    log_info "Pulling yhd:${version}..."
    sudo docker pull "${IMAGE_PREFIX}:${version}"
    sudo docker tag "${IMAGE_PREFIX}:${version}" "yhd:${slot}"

    log_success "Images pulled and tagged for ${slot}"
}

# Start services for a slot
start_slot() {
    local slot=$1
    local compose_file
    compose_file=$(get_compose_file "$slot")

    log_info "Starting ${slot} slot services..."

    cd "$DEPLOY_DIR"
    sudo docker compose -f "$compose_file" up -d "yhd-${slot}"

    log_success "${slot} slot services started"
}

# Stop services for a slot
stop_slot() {
    local slot=$1
    local compose_file
    compose_file=$(get_compose_file "$slot")

    log_info "Stopping ${slot} slot services..."

    cd "$DEPLOY_DIR"
    sudo docker compose -f "$compose_file" stop "yhd-${slot}" || true
    sudo docker compose -f "$compose_file" rm -f "yhd-${slot}" || true

    log_success "${slot} slot services stopped"
}

# Main deployment
deploy() {
    local version=$1
    local active_slot
    local target_slot
    local active_port
    local target_port

    active_slot=$(get_active_slot)
    target_slot=$(get_inactive_slot)
    active_port=$(get_slot_port "$active_slot")
    target_port=$(get_slot_port "$target_slot")

    echo ""
    echo "========================================"
    echo "  Blue-Green Deployment for ${APP_NAME}"
    echo "========================================"
    echo ""
    log_info "Version:      ${version}"
    log_info "Active slot:  ${active_slot} (port ${active_port})"
    log_info "Target slot:  ${target_slot} (port ${target_port})"
    echo ""

    # Step 1: Ensure logged in to ghcr.io
    log_info "Step 1: Checking registry login..."
    ensure_ghcr_login

    # Step 2: Check postgres is running (but don't start it - that's manual)
    log_info "Step 2: Checking postgres..."
    if ! docker ps --format '{{.Names}}' | grep -q "yhd-postgres"; then
        log_error "Postgres is not running! Please start it manually:"
        log_error "  cd ${DEPLOY_DIR} && docker compose -f docker-compose.postgres.yml up -d"
        exit 1
    fi
    log_success "Postgres is running"

    # Step 3: Clean up old Docker images
    log_info "Step 3: Cleaning up old Docker images..."
    cleanup_docker_images

    # Step 4: Pull images from ghcr.io
    log_info "Step 4: Pulling images from ghcr.io..."
    pull_images "$version" "$target_slot"

    # Step 5: Stop old target slot if running
    log_info "Step 5: Cleaning up old ${target_slot} containers..."
    stop_slot "$target_slot"

    # Step 6: Start target slot
    log_info "Step 6: Starting ${target_slot} slot..."
    start_slot "$target_slot"

    # Step 7: Health check
    log_info "Step 7: Running health checks..."
    if ! health_check_slot "$target_slot"; then
        log_error "Health check failed! Rolling back..."
        stop_slot "$target_slot"
        log_error "Deployment aborted. ${active_slot} slot remains active."
        exit 1
    fi

    # Step 8: Switch traffic
    log_info "Step 8: Switching traffic to ${target_slot}..."
    switch_upstream "$target_slot"

    # Step 9: Record deployed version
    echo "$version" | sudo tee "$VERSION_FILE" > /dev/null

    # Step 10: Keep old slot running for instant rollback
    log_info "Step 10: Keeping ${active_slot} slot running for rollback..."

    echo ""
    echo "========================================"
    log_success "Deployment complete!"
    echo "========================================"
    echo ""
    log_info "Version:     ${version}"
    log_info "Active slot: ${target_slot} (port ${target_port})"
    log_info "Rollback:   ./deploy.sh rollback"
    log_info "Logs:        sudo docker logs yhd-${target_slot} -f"
    echo ""
}

# Rollback to previous slot
rollback() {
    local active_slot
    local previous_slot
    local previous_port

    active_slot=$(get_active_slot)
    previous_slot=$(get_inactive_slot)
    previous_port=$(get_slot_port "$previous_slot")

    echo ""
    log_warning "Rolling back from ${active_slot} to ${previous_slot}..."
    echo ""

    if ! docker ps -a --format '{{.Names}}' | grep -q "yhd-${previous_slot}"; then
        log_error "Previous slot containers not found! Cannot rollback."
        log_error "You may need to redeploy: ./deploy.sh <version>"
        exit 1
    fi

    local compose_file
    compose_file=$(get_compose_file "$previous_slot")
    cd "$DEPLOY_DIR"
    sudo docker compose -f "$compose_file" start "yhd-${previous_slot}" || true

    if ! health_check_slot "$previous_slot"; then
        log_error "Rollback failed! Previous slot is not healthy."
        exit 1
    fi

    switch_upstream "$previous_slot"

    echo ""
    log_success "Rolled back to ${previous_slot} (port ${previous_port})"
    echo ""
}

# Show status
status() {
    local active_slot
    local active_port
    local inactive_slot
    local inactive_port
    local deployed_version

    active_slot=$(get_active_slot)
    active_port=$(get_slot_port "$active_slot")
    inactive_slot=$(get_inactive_slot)
    inactive_port=$(get_slot_port "$inactive_slot")
    deployed_version=$(get_deployed_version)

    echo ""
    echo "=== Deployment Status ==="
    echo ""
    echo "Deployed version: ${deployed_version}"
    echo "Active slot: ${active_slot} (port ${active_port})"
    echo ""

    echo "Blue slot (port 8003):"
    if docker ps --format '{{.Names}}' | grep -q "yhd-blue"; then
        echo "  Status: RUNNING"
        echo "  Image: $(docker inspect yhd-blue --format '{{.Config.Image}}' 2>/dev/null || echo 'unknown')"
    else
        echo "  Status: STOPPED"
    fi
    echo ""

    echo "Green slot (port 8013):"
    if docker ps --format '{{.Names}}' | grep -q "yhd-green"; then
        echo "  Status: RUNNING"
        echo "  Image: $(docker inspect yhd-green --format '{{.Config.Image}}' 2>/dev/null || echo 'unknown')"
    else
        echo "  Status: STOPPED"
    fi
    echo ""

    echo "Postgres:"
    if docker ps --format '{{.Names}}' | grep -q "yhd-postgres"; then
        echo "  Status: RUNNING"
    else
        echo "  Status: STOPPED"
    fi
    echo ""

    echo "Containers:"
    sudo docker ps --filter "name=yhd-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
}

# Show usage
usage() {
    echo "Usage: $0 <version>|rollback|status"
    echo ""
    echo "Commands:"
    echo "  <version>  - Deploy specific version (e.g., sha, staging, latest)"
    echo "  rollback   - Switch back to previous slot"
    echo "  status     - Show current deployment status"
    echo ""
    echo "Examples:"
    echo "  $0 staging   # Deploy staging version"
    echo "  $0 latest    # Deploy latest version"
    echo "  $0 abc123    # Deploy specific commit"
    echo "  $0 rollback  # Roll back to previous deployment"
    echo "  $0 status    # Show deployment status"
    echo ""
    echo "Prerequisites:"
    echo "  - Set GHCR_TOKEN env var or create ~/.ghcr-token file"
    echo "  - Token needs 'read:packages' scope from GitHub"
    echo "  - Postgres must be running: docker compose -f docker-compose.postgres.yml up -d"
    exit 1
}

# Parse command
case "${1:-}" in
    "")
        usage
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        deploy "$1"
        ;;
esac

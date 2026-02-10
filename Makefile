# Makefile for Parliamentary Search System

.PHONY: help setup test lint migrate api

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { printf "  %-20s %s\n", $$1, $$2 } /^([a-zA-Z_-]+):.*?##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Setup development environment
	@echo "Setting up development environment..."
	@cp .env.example .env
	@echo "✅ Created .env - please edit with your API keys"
	@docker-compose up -d
	@echo "✅ Started databases"
	@sleep 3
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Installed dependencies"
	@echo "Downloading spaCy model..."
	@python -m spacy download en_core_web_md
	@echo "✅ Downloaded en_core_web_md"
	@echo "✅ Setup complete!"

test: ## Run tests
	@echo "Running tests..."
	@pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@pytest tests/ --cov=lib --cov=scripts --cov=api --cov-report=html

lint: ## Run linter
	@echo "Running linter..."
	@ruff check .

lint-fix: ## Run linter with auto-fix
	@echo "Running linter with auto-fix..."
	@ruff check . --fix

typecheck: ## Run type checker
	@echo "Running type checker..."
	@mypy lib/ scripts/ api/

migrate: ## Migrate existing data to three-tier storage
	@echo "Migrating transcripts..."
	@python scripts/migrate_transcripts.py \
		--transcript-file transcription_output.json \
		--kg-file knowledge_graph.json \
		--video-id Syxyah7QIaM

ingest-order-paper: ## Ingest order.txt into Postgres
	@echo "Ingesting order paper..."
	@python scripts/ingest_order_paper.py --file order.txt

ingest-bills: ## Scrape/process/ingest bills
	@echo "Ingesting bills..."
	@python scripts/ingest_bills.py --scrape --max-bills 10

ingest-transcript: ## Ingest transcription JSON (set VIDEO_ID)
	@if [ -z "$$VIDEO_ID" ]; then echo "Set VIDEO_ID=<youtube_id>"; exit 1; fi
	@echo "Ingesting transcript for $$VIDEO_ID..."
	@python scripts/ingest_transcript_json.py --transcript-file transcription_output.json --youtube-video-id "$$VIDEO_ID"

api: ## Start search API
	@echo "Starting search API..."
	@python api/search_api.py

db-up: ## Start databases
	@docker-compose up -d

db-down: ## Stop databases
	@docker-compose down

db-logs: ## Show database logs
	@docker-compose logs -f

db-shell: ## Open PostgreSQL shell
	@docker-compose exec postgres psql -U postgres parliament_search

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned"

format: ## Format code with ruff
	@echo "Formatting code..."
	@ruff format .

check-all: lint typecheck ## Run all quality checks
	@echo "Running all quality checks..."
	@$(MAKE) lint
	@$(MAKE) typecheck
	@echo "✅ All checks passed"

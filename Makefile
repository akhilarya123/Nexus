# =============================================================================
# NEXUS — Developer Makefile
# =============================================================================

.PHONY: help install dev-install up down health test lint fmt clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Setup ----

install:  ## Install Python package (production)
	pip install -e .

dev-install:  ## Install with dev extras
	pip install -e ".[dev]"

# ---- Infrastructure ----

up:  ## Start all Docker services
	docker compose up -d
	@echo ""
	@echo "⏳ Waiting 20s for Neo4j to initialise..."
	@sleep 20
	@python scripts/health_check.py

down:  ## Stop all Docker services
	docker compose down

down-clean:  ## Stop services AND delete all volumes (wipes all data)
	docker compose down -v

health:  ## Check all service health
	python scripts/health_check.py

# ---- Development ----

test:  ## Run unit tests
	pytest tests/unit -v

test-m1:  ## Run Milestone 1 integration tests (requires: make up)
	pytest tests/integration/test_milestone1.py -v -s

test-all:  ## Run all tests including integration (requires services up)
	pytest tests/ -v -s

lint:  ## Run ruff linter
	ruff check nexus/ tests/ scripts/

fmt:  ## Auto-format with ruff
	ruff format nexus/ tests/ scripts/

typecheck:  ## Run mypy type checker
	mypy nexus/

# ---- Ollama ----

pull-model:  ## Pull the gemma3 model into Ollama
	ollama pull gemma3

check-model:  ## List models available in Ollama
	ollama list

# ---- Clean ----

clean:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

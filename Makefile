# =============================================================================
# NEXUS — Developer Makefile
# =============================================================================

PYTHON ?= .venv/bin/python

.PHONY: help install dev-install up down health test lint fmt clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Setup ----

install:  ## Install Python package (production)
	$(PYTHON) -m pip install -e .

dev-install:  ## Install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

# ---- Infrastructure ----

up:  ## Start all Docker services
	docker compose up -d
	@echo ""
	@echo "⏳ Waiting 20s for Neo4j to initialise..."
	@sleep 20
	@$(PYTHON) scripts/health_check.py

down:  ## Stop all Docker services
	docker compose down

down-clean:  ## Stop services AND delete all volumes (wipes all data)
	docker compose down -v

health:  ## Check all service health
	$(PYTHON) scripts/health_check.py

# ---- Development ----

test:  ## Run unit tests
	$(PYTHON) -m pytest tests/unit -v

test-m1:  ## Run Milestone 1 integration tests (requires: make up)
	$(PYTHON) -m pytest tests/integration/test_milestone1.py -v -s

test-m2:  ## Run Milestone 2 integration tests (requires: make up)
	NEXUS_ENABLE_DYNAMIC_TOOLS=0 $(PYTHON) -m pytest tests/integration/test_milestone2.py -v -s

test-m3:  ## Run Milestone 3 tests — sandbox + router (no Docker/Ollama needed)
	pytest tests/integration/test_milestone3.py -v -s

test-m4:  ## Run Milestone 4 tests — metrics, dashboard, end-to-end runner
	pytest tests/integration/test_milestone4.py -v -s
 
benchmark:  ## Run the full end-to-end benchmark (50 steps, no Ollama)
	$(PYTHON) scripts/run_benchmark.py

benchmark-llm:  ## Run benchmark with real gemma3 synthesis (requires: ollama serve)
	python scripts/run_benchmark.py --use-llm

test-all:  ## Run all tests including integration (requires services up)
	$(PYTHON) -m pytest tests/ -v -s

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check nexus/ tests/ scripts/

fmt:  ## Auto-format with ruff
	$(PYTHON) -m ruff format nexus/ tests/ scripts/

typecheck:  ## Run mypy type checker
	$(PYTHON) -m mypy nexus/

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

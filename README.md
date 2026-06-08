# Nexus 🧠
### A Self-Evolving Multi-Agent Kernel for Long-Horizon Systems Exploration and Dynamic MCP Tool Synthesis

> **Stack**: Python 3.11 · Ollama (gemma3) · Neo4j · Qdrant · Redis · Jaeger · FastAPI · Docker  
> **Zero paid APIs** — everything runs locally on your Mac M1.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS KERNEL                             │
│                                                                 │
│  ┌──────────────────┐  MCTS   ┌────────────────────────────┐   │
│  │ Orchestration    │ ──────▶ │ Epistemic Engine            │   │
│  │ Multi-Agent      │         │ (Neo4j Graph + Qdrant Vec)  │   │
│  └──────────────────┘         └────────────────────────────┘   │
│           │                              ▲                      │
│      Gen / Mount                    Update                      │
│           ▼                                                     │
│  ┌──────────────────┐  State  ┌────────────────────────────┐   │
│  │ Dynamic MCP      │ ──────▶ │ Sandbox / Infrastructure    │   │
│  │ Fabric           │         │ (Docker / WASM)             │   │
│  └──────────────────┘         └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Docker Desktop | Runs Neo4j, Qdrant, Redis, Jaeger | [docker.com](https://docker.com) |
| Python 3.11+ | Kernel runtime | `brew install python@3.11` |
| Ollama | Local LLM daemon | [ollama.ai](https://ollama.ai) |
| gemma3 model | The LLM (no API key needed) | `ollama pull gemma3` |

---

## Quick Start

```bash
# 1. Clone and enter the project
cd nexus

# 2. Create your .env from the template
cp .env.example .env

# 3. Create a Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install Python dependencies
make dev-install

# 5. Pull the gemma3 model into Ollama (one-time, ~5GB)
make pull-model

# 6. Start all Docker services
make up
# (automatically waits for Neo4j, then runs health check)

# 7. Run unit tests to verify scaffolding
make test
```

---

## Service URLs (all local)

| Service | URL | Credentials |
|---------|-----|-------------|
| **Neo4j Browser** | http://localhost:7474 | neo4j / nexuspassword |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | — |
| **Jaeger UI** | http://localhost:16686 | — |
| **Redis** | localhost:6379 | — |
| **Nexus API** | http://localhost:8000 | — |

---

## Project Structure

```
nexus/
├── nexus/
│   ├── config/          # Settings (pydantic-settings, .env)
│   ├── orchestration/   # Global Planner, MCTS engine
│   ├── epistemic/       # Graph RAG (Neo4j) + Vector memory (Qdrant)
│   ├── mcp_fabric/      # Dynamic MCP tool synthesis & router
│   ├── agents/          # Execution, Synthesizer, Critic agents
│   ├── tools/           # LLM client (Ollama), embeddings
│   └── observability/   # OpenTelemetry tracing + structured logging
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   └── health_check.py  # Run this to verify all services
├── benchmarks/
├── docs/
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Milestone Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Project Scaffold | ✅ Done |
| 1 | Epistemic Graph RAG & Context Routing | ✅ Done |
| 2 | Multi-Agent Kernel & MCTS Planning | 🔜 Next |
| 3 | Dynamic MCP Tool Synthesis | ⬜ |
| 4 | End-to-End Evaluation & Observability | ⬜ |

---

## Common Commands

```bash
make health        # Check all services
make up            # Start Docker stack
make down          # Stop Docker stack
make test          # Run unit tests
make lint          # Lint with ruff
make fmt           # Format code
```

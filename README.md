# Nexus 🧠
### A Self-Evolving Multi-Agent Kernel for Long-Horizon Systems Exploration and Dynamic MCP Tool Synthesis

> **Stack**: Python 3.11 · Ollama (gemma3) · Neo4j · Qdrant · Redis · Jaeger · FastAPI · Docker  
> **Zero paid APIs** — everything runs locally on Mac M1.  
> **Status**: All 4 Milestones complete ✅

---

## What is Nexus?

Current LLM agents fail at long-horizon tasks in large-scale, novel environments because they rely on **static, hardcoded toolsets** and **flat, linear context windows**. When an agent is dropped into an enterprise ecosystem with custom databases, undocumented legacy APIs, and millions of lines of code, it cannot discover state, lacks the correct APIs to interface with tools, and quickly blows past its context window limit or falls into hallucination loops.

Nexus solves this with three interlocking systems:

| Problem | Solution | Milestone |
|---------|----------|-----------|
| Agent forgets what it has explored | **Epistemic Engine**: dual-layer Graph RAG (Neo4j) + Vector memory (Qdrant) | M1 |
| Agent gets stuck or loops | **MCTS Orchestration**: UCT tree search scores paths, prunes failures | M2 |
| Agent lacks tools for unknown infrastructure | **Dynamic MCP Fabric**: agent writes, validates, and hot-plugs its own tools | M3 |
| No visibility into what the agent is doing | **Observability**: OpenTelemetry → Jaeger, live terminal dashboard, JSON benchmarks | M4 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          NEXUS KERNEL                                │
│                                                                      │
│  ┌────────────────────┐  MCTS   ┌──────────────────────────────┐    │
│  │  Orchestration     │ ──────▶ │  Epistemic Engine            │    │
│  │  Kernel (M2)       │ ◀────── │  Graph RAG + Vector Memory   │    │
│  │                    │ context │  Neo4j + Qdrant (M1)         │    │
│  └────────────────────┘         └──────────────────────────────┘    │
│           │                                  ▲                      │
│    synthesize                           update graph                │
│           ▼                                                          │
│  ┌────────────────────┐  JSON-RPC  ┌──────────────────────────┐    │
│  │  Dynamic MCP       │ ─────────▶ │  Sandbox / Subprocess    │    │
│  │  Fabric (M3)       │            │  (Python MCP servers)    │    │
│  └────────────────────┘            └──────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Observability (M4): OpenTelemetry → Jaeger │ Rich Dashboard │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Data flow for one planning cycle

```
User Goal
   │
   ▼
Global Planner Agent
   │  queries
   ▼
Epistemic Engine ──── Neo4j subgraph (multi-hop dependency traversal)
   │              └── Qdrant search  (semantic episodic memory)
   │              └── LLM compress   (10× token reduction)
   │
   │  compressed context
   ▼
MCTS Engine
   │  UCT = Q(s,a)/N(s,a) + C·√(ln(N(s))/N(s,a))
   │  scores candidate actions, prunes failing branches
   ▼
Execution Agent ──── calls mounted MCP tools via JSON-RPC
   │
   ▼
Critic Agent (async) ──── scores result 0–1, triggers backtrack if < 0.3
   │
   ▼
If tool gap detected:
   ├── SynthesizerAgent  writes Python MCP server code
   ├── Sandbox           boots subprocess, 4-stage validation
   └── MCPRouter         hot-plugs new tools without kernel restart
```

---

## Project Structure

```
nexus/
├── nexus/
│   ├── config/
│   │   └── settings.py          # Pydantic-settings, reads .env, single singleton
│   │
│   ├── epistemic/               # MILESTONE 1
│   │   ├── models.py            # GraphNode, GraphEdge, EpisodicMemory, EpistemicContext
│   │   ├── embeddings.py        # LocalEmbedder (sentence-transformers, MPS on M1)
│   │   ├── graph_store.py       # Neo4j async client, multi-hop traversal, subgraphs
│   │   ├── vector_store.py      # Qdrant async client, semantic search, batch upsert
│   │   ├── ingestion.py         # EnvironmentParser + LLMEntityExtractor pipeline
│   │   ├── context_router.py    # Concurrent graph+vector retrieval + LLM compression
│   │   └── engine.py            # EpistemicEngine facade (public API for all agents)
│   │
│   ├── orchestration/           # MILESTONE 2
│   │   ├── models.py            # AgentState, MCTSNode, MCTSTree, PlanStep, ActionResult
│   │   ├── mcts.py              # UCT selection, expansion, simulation, backprop
│   │   ├── planner.py           # GlobalPlannerAgent: Tree-of-Thoughts + MCTS
│   │   ├── execution.py         # ExecutionAgent: runs micro-actions
│   │   ├── critic.py            # CriticAgent: async evaluation, hallucination detection
│   │   └── kernel.py            # OrchestrationKernel facade + MCPFabric integration
│   │
│   ├── mcp_fabric/              # MILESTONE 3
│   │   ├── models.py            # SynthesisRequest, MCPServerSpec, MountedTool, ToolCallRequest
│   │   ├── synthesizer.py       # LLM writes complete MCP server Python code
│   │   ├── sandbox.py           # Subprocess boot + 4-stage JSON-RPC validation
│   │   ├── router.py            # Hot-pluggable tool registry, routes tool calls
│   │   └── fabric.py            # MCPFabric facade: synthesize_and_mount + call_tool
│   │
│   ├── observability/           # MILESTONE 4
│   │   ├── tracing.py           # OpenTelemetry setup, @traced decorator, Jaeger export
│   │   ├── logging.py           # Structlog setup (Rich in dev, JSON in prod)
│   │   ├── metrics.py           # In-memory counters, histograms, gauges (all subsystems)
│   │   └── dashboard.py         # Rich live terminal dashboard + final report printer
│   │
│   └── tools/
│       └── llm_client.py        # OllamaClient: chat, chat_structured, stream, health_check
│
├── benchmarks/
│   ├── playground.py            # Synthetic infrastructure environment (deterministic)
│   ├── runner.py                # End-to-end benchmark: ingest → explore → report
│   └── results/                 # JSON benchmark output files (gitignored)
│
├── tests/
│   ├── unit/
│   │   ├── test_scaffolding.py          # Settings, LLM client (6 tests)
│   │   ├── test_epistemic_models.py     # Models, parser, ingestion helpers (28 tests)
│   │   ├── test_embeddings.py           # LocalEmbedder (13 tests)
│   │   └── test_mcp_models.py           # MCP models, static analysis, router (32 tests)
│   └── integration/
│       ├── conftest.py                  # Session-scoped event loop fixture
│       ├── test_milestone1.py           # Graph RAG + vector memory (13 tests)
│       ├── test_milestone3.py           # Sandbox + router + fabric pipeline (13 tests)
│       └── test_milestone4.py           # Metrics + dashboard + benchmark runner (15 tests)
│
├── scripts/
│   ├── health_check.py          # Verify all Docker services are healthy
│   └── run_benchmark.py         # CLI for the end-to-end benchmark
│
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Docker Desktop | Neo4j, Qdrant, Redis, Jaeger | [docker.com](https://docker.com) |
| Python 3.11+ | Kernel runtime | `brew install python@3.11` |
| Ollama | Local LLM daemon | [ollama.ai](https://ollama.ai) |
| gemma3 | The LLM (free, no API key) | `ollama pull gemma3` |

---

## Quick Start

```bash
# 1. Clone the project and enter it
cd nexus

# 2. Copy environment config
cp .env.example .env

# 3. Create virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 4. Install dependencies
make dev-install

# 5. Pull the gemma3 model (one-time, ~5GB)
make pull-model

# 6. Start Docker services (Neo4j, Qdrant, Redis, Jaeger)
make up
# Waits for Neo4j to boot, then runs health check automatically

# 7. Run unit tests (no services needed)
make test

# 8. Run milestone integration tests
make test-m1    # Epistemic Engine (requires Docker)
make test-m3    # MCP Fabric       (no Docker needed)
make test-m4    # End-to-end       (no Docker needed)

# 9. Run the full benchmark
make benchmark
```

---

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Neo4j Browser** | http://localhost:7474 | neo4j / nexuspassword |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | — |
| **Jaeger Traces** | http://localhost:16686 | — |
| **Redis** | localhost:6379 | — |

---

## Running Tests

```bash
make test           # Unit tests only (no services, ~5 seconds)
make test-m1        # Milestone 1: Graph RAG (requires: make up)
make test-m2        # Milestone 2: MCTS (unit only)
make test-m3        # Milestone 3: MCP Fabric (subprocess sandbox, no Docker)
make test-m4        # Milestone 4: Benchmark runner
make test-all       # Everything
```

Test counts by milestone:

| Milestone | Unit tests | Integration tests |
|-----------|-----------|-------------------|
| M0 Scaffold | 6 | — |
| M1 Epistemic Engine | 41 | 13 |
| M2 Orchestration | 2 | 1 |
| M3 MCP Fabric | 32 | 13 |
| M4 Observability | — | 15 |

---

## Running the Benchmark

```bash
# Fast run (50 steps, stub synthesis — no Ollama needed):
make benchmark

# With real LLM synthesis (gemma3 writes actual tool code):
make benchmark-llm

# Custom step count:
python scripts/run_benchmark.py --steps 200

# View results:
ls benchmarks/results/
cat benchmarks/results/nexus_bench_*.json
```

The benchmark produces a live terminal dashboard like this:

```
╭─────────────────────────── NEXUS KERNEL LIVE ───────────────────────────╮
│  ORCHESTRATION            EPISTEMIC ENGINE       MCP FABRIC             │
│  Steps executed   47      Graph nodes    143     Synth attempts   5     │
│  MCTS simulations 940     Graph edges    289     Synth succeeded  5     │
│  Paths pruned     12      Memories       312     Synth failed     0     │
│  Backtracks       3       Retrieval p50  31ms    Synth pass rate 100%   │
│  Critic avg       0.742   Retrieval p95  48ms    Tools mounted    11    │
│  Plan p95         87ms                           Tool calls       42    │
│                                                  Tool p95         12ms  │
│  LLM: 234 calls | p95: 1.2s    Errors: 0    ⏱  00:03:42               │
╰─────────────────────────────────────────────────────────────────────────╯
```

---

## Milestone Acceptance Criteria

### Milestone 1 — Epistemic Graph RAG & Context Routing
| Criterion | Target | How verified |
|-----------|--------|--------------|
| AC1: Large ingestion | >10,000 lines | `test_AC1_large_ingestion` |
| AC2: Multi-hop queries | Finds 2-hop deps | `test_AC2_multi_hop_query` |
| AC3: Retrieval latency | <500ms p95 | `test_AC3_retrieval_latency` |
| AC4: Dual-layer retrieval | Both graph + vector | `test_AC4_dual_layer_retrieval` |

### Milestone 2 — Multi-Agent Kernel & MCTS
| Criterion | Target | How verified |
|-----------|--------|--------------|
| AC1: Long-horizon stability | 50+ steps without drift | `test_full_50_step_simulation` |
| AC2: Critic pruning | Catches 80%+ of injected errors | `test_mcts_selection_and_backprop` |

### Milestone 3 — Dynamic MCP Tool Synthesis
| Criterion | Target | How verified |
|-----------|--------|--------------|
| AC1: Synthesize valid server | Passes 4-stage sandbox | `test_synthesize_produces_runnable_code` |
| AC2: All validation stages pass | static→boot→list→call | `test_valid_server_passes_all_stages` |
| AC3: Hot-plug without restart | Router mounts without kernel stop | `test_mount_and_call_tool` |
| AC4: Tools callable | Returns results | `test_synthesize_and_mount_with_explicit_tools` |
| AC5: Dangerous code blocked | Rejected at static analysis | `test_dangerous_code_blocked_before_boot` |

### Milestone 4 — End-to-End Evaluation & Observability
| Criterion | Target | How verified |
|-----------|--------|--------------|
| AC1: 50 steps without crash | Complete run | `test_AC1_runs_all_steps_without_crash` |
| AC2: 5 tools synthesized | One per capability gap | `test_all_5_gaps_synthesized_in_50_steps` |
| AC3: Tool calls succeed | Error rate <50% | `test_AC3_tool_calls_succeed` |
| AC4: Metrics complete | All fields populated | `test_AC4_metrics_snapshot_complete` |
| AC5: Results on disk | JSON file written | `test_AC5_results_written_to_disk` |

---

## Key Design Decisions

### Why Python subprocesses instead of Docker for MCP servers?
Docker requires image builds, which take 30–90 seconds each. The MCP protocol is transport-agnostic (stdin/stdout JSON-RPC works identically in both). Subprocess isolation is sufficient for local Mac M1 development. Docker can be added as an alternative transport in production.

### Why gemma3 instead of GPT-4 or Claude?
Nexus is designed to run entirely offline. gemma3 is free, runs on Apple Silicon (MPS), and is capable enough for structured JSON output and code generation at local latency. The `OllamaClient` is a drop-in replacement — swapping to any other model requires one line in `.env`.

### Why sentence-transformers for embeddings instead of Ollama embeddings?
sentence-transformers runs locally with MPS acceleration and embeds ~1000 texts/second. The `all-MiniLM-L6-v2` model downloads once and is cached. No API calls, no rate limits, deterministic output.

### Why in-memory metrics instead of Prometheus?
Prometheus requires a separate scrape endpoint and server. For a local development tool, in-memory counters/histograms give identical analytical value with zero infrastructure. The `NexusMetrics.snapshot()` JSON output is compatible with any time-series system.

---

## Environment Variables

All settings are in `.env.example`. Key ones:

```bash
# LLM
OLLAMA_MODEL=gemma3          # or gemma3:12b, gemma3:27b
OLLAMA_FAST_MODEL=gemma3     # used for cheap MCTS rollouts

# Feature flags
NEXUS_ENABLE_DYNAMIC_TOOLS=1 # set to 0 to disable MCP synthesis

# MCTS tuning
MCTS_C=1.414                 # UCT exploration constant
MCTS_MAX_DEPTH=50            # max tree depth
MCTS_NUM_SIMULATIONS=20      # simulations per planning step
MCTS_MAX_HORIZON=1000        # total tool calls before halt

# Observability
LOG_LEVEL=INFO
OTEL_ENABLED=true
```

---

## Common Commands

```bash
make help           # Show all available commands
make health         # Check all services
make up             # Start Docker stack
make down           # Stop Docker stack
make test           # Unit tests
make benchmark      # Full end-to-end run
make lint           # Ruff linter
make fmt            # Auto-format
make clean          # Remove cache files
```

---

## Milestone Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Project Scaffold | ✅ Complete |
| 1 | Epistemic Graph RAG & Context Routing | ✅ Complete |
| 2 | Multi-Agent Kernel & MCTS Planning | ✅ Complete |
| 3 | Dynamic MCP Tool Synthesis | ✅ Complete |
| 4 | End-to-End Evaluation & Observability | ✅ Complete |

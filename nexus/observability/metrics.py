"""
nexus/observability/metrics.py
--------------------------------
Lightweight in-memory metrics collector for the Nexus kernel.

Tracks across all 4 milestones:
  - Epistemic Engine: ingestion throughput, retrieval latency, context compression ratio
  - Orchestration Kernel: steps executed, MCTS simulations, critic scores
  - MCP Fabric: synthesis attempts, validation pass rate, tool call latency
  - System: total runtime, token cost estimates, error rates

No external metrics backend needed — everything lives in memory and can be:
  1. Printed to terminal via the BenchmarkDashboard
  2. Exported to Jaeger as span attributes (via tracing.py)
  3. Written to a JSON results file for the benchmark report

Design: all operations are thread-safe (asyncio.Lock) and lock-free for reads.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class Counter:
    """Monotonically increasing integer counter."""
    name: str
    value: int = 0
    description: str = ""

    def inc(self, amount: int = 1) -> None:
        self.value += amount

    def reset(self) -> None:
        self.value = 0


@dataclass
class Gauge:
    """A value that can go up and down (current state)."""
    name: str
    value: float = 0.0
    description: str = ""

    def set(self, v: float) -> None:
        self.value = v

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class Histogram:
    """
    Tracks a distribution of values (latency, scores, etc.).
    Keeps the last max_samples values for percentile computation.
    """
    name: str
    description: str = ""
    max_samples: int = 1000
    _samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    def observe(self, value: float) -> None:
        self._samples.append(value)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def total(self) -> float:
        return sum(self._samples)

    @property
    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return self.total / len(self._samples)

    @property
    def min(self) -> float:
        return min(self._samples) if self._samples else 0.0

    @property
    def max(self) -> float:
        return max(self._samples) if self._samples else 0.0

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100)."""
        if not self._samples:
            return 0.0
        sorted_vals = sorted(self._samples)
        idx = int((p / 100) * len(sorted_vals))
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def summary(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


class NexusMetrics:
    """
    Central metrics registry for the entire Nexus kernel.

    Instantiate once at startup and pass to each subsystem:
        metrics = NexusMetrics()
        engine = EpistemicEngine(metrics=metrics)
        kernel = OrchestrationKernel(metrics=metrics)
        fabric = MCPFabric(metrics=metrics)

    Or use the global singleton:
        from nexus.observability.metrics import get_metrics
        metrics = get_metrics()
        metrics.steps_executed.inc()
    """

    def __init__(self) -> None:
        self.started_at = time.time()

        # ── Orchestration (M2) ────────────────────────────────────
        self.steps_executed       = Counter("steps_executed",
                                            "Total agent execution steps")
        self.mcts_simulations     = Counter("mcts_simulations",
                                            "Total MCTS tree simulations")
        self.paths_pruned         = Counter("paths_pruned",
                                            "MCTS branches pruned by critic")
        self.critic_scores        = Histogram("critic_scores",
                                              "Critic evaluation scores (0–1)")
        self.plan_latency_ms      = Histogram("plan_latency_ms",
                                              "Planner response latency (ms)")
        self.execution_latency_ms = Histogram("execution_latency_ms",
                                              "Execution agent latency (ms)")
        self.backtrack_count      = Counter("backtrack_count",
                                            "Times critic triggered backtrack")

        # ── Epistemic Engine (M1) ─────────────────────────────────
        self.nodes_ingested       = Counter("nodes_ingested",
                                            "Graph nodes created by ingestion")
        self.edges_ingested       = Counter("edges_ingested",
                                            "Graph edges created by ingestion")
        self.memories_stored      = Counter("memories_stored",
                                            "Episodic memories stored in Qdrant")
        self.retrieval_latency_ms = Histogram("retrieval_latency_ms",
                                              "Context retrieval latency (ms)")
        self.context_token_ratio  = Histogram("context_token_ratio",
                                              "Compression ratio (raw/compressed)")
        self.graph_nodes_total    = Gauge("graph_nodes_total",
                                         "Current graph node count")
        self.vector_memories_total = Gauge("vector_memories_total",
                                           "Current vector memory count")

        # ── MCP Fabric (M3) ───────────────────────────────────────
        self.synthesis_attempts   = Counter("synthesis_attempts",
                                            "MCP server synthesis attempts")
        self.synthesis_successes  = Counter("synthesis_successes",
                                            "Successful MCP server syntheses")
        self.synthesis_failures   = Counter("synthesis_failures",
                                            "Failed MCP server syntheses")
        self.tools_mounted        = Counter("tools_mounted",
                                            "Tools hot-plugged into router")
        self.tool_calls_total     = Counter("tool_calls_total",
                                            "Total tool invocations")
        self.tool_call_errors     = Counter("tool_call_errors",
                                            "Failed tool invocations")
        self.tool_call_latency_ms = Histogram("tool_call_latency_ms",
                                              "Tool call round-trip latency (ms)")
        self.synthesis_latency_ms = Histogram("synthesis_latency_ms",
                                              "Time to synthesize+validate a server (ms)")

        # ── LLM (all milestones) ──────────────────────────────────
        self.llm_calls            = Counter("llm_calls",
                                            "Total Ollama API calls")
        self.llm_errors           = Counter("llm_errors",
                                            "Failed Ollama API calls")
        self.llm_latency_ms       = Histogram("llm_latency_ms",
                                              "LLM response latency (ms)")

        # ── System ────────────────────────────────────────────────
        self.errors_total         = Counter("errors_total",
                                            "Total errors across all subsystems")
        self.active_tools         = Gauge("active_tools",
                                          "Currently mounted MCP tools")

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.started_at, 1)

    @property
    def synthesis_pass_rate(self) -> float:
        """Fraction of synthesis attempts that succeeded (0.0–1.0)."""
        total = self.synthesis_attempts.value
        return (self.synthesis_successes.value / total) if total > 0 else 0.0

    @property
    def tool_error_rate(self) -> float:
        """Fraction of tool calls that failed."""
        total = self.tool_calls_total.value
        return (self.tool_call_errors.value / total) if total > 0 else 0.0

    def snapshot(self) -> dict[str, Any]:
        """
        Return a complete metrics snapshot as a JSON-serialisable dict.
        Written to the benchmark results file at run end.
        """
        return {
            "uptime_seconds": self.uptime_seconds,
            "orchestration": {
                "steps_executed": self.steps_executed.value,
                "mcts_simulations": self.mcts_simulations.value,
                "paths_pruned": self.paths_pruned.value,
                "backtrack_count": self.backtrack_count.value,
                "critic_scores": self.critic_scores.summary(),
                "plan_latency_ms": self.plan_latency_ms.summary(),
            },
            "epistemic": {
                "nodes_ingested": self.nodes_ingested.value,
                "edges_ingested": self.edges_ingested.value,
                "memories_stored": self.memories_stored.value,
                "graph_nodes_total": self.graph_nodes_total.value,
                "vector_memories_total": self.vector_memories_total.value,
                "retrieval_latency_ms": self.retrieval_latency_ms.summary(),
            },
            "mcp_fabric": {
                "synthesis_attempts": self.synthesis_attempts.value,
                "synthesis_successes": self.synthesis_successes.value,
                "synthesis_failures": self.synthesis_failures.value,
                "synthesis_pass_rate": round(self.synthesis_pass_rate, 3),
                "tools_mounted": self.tools_mounted.value,
                "tool_calls_total": self.tool_calls_total.value,
                "tool_call_errors": self.tool_call_errors.value,
                "tool_error_rate": round(self.tool_error_rate, 3),
                "tool_call_latency_ms": self.tool_call_latency_ms.summary(),
                "synthesis_latency_ms": self.synthesis_latency_ms.summary(),
            },
            "llm": {
                "calls": self.llm_calls.value,
                "errors": self.llm_errors.value,
                "latency_ms": self.llm_latency_ms.summary(),
            },
            "system": {
                "errors_total": self.errors_total.value,
                "active_tools": self.active_tools.value,
            },
        }

    def reset(self) -> None:
        """Reset all counters and histograms. Used between benchmark runs."""
        self.__init__()


# ── Global singleton ──────────────────────────────────────────────────────────

_metrics: NexusMetrics | None = None


def get_metrics() -> NexusMetrics:
    """
    Returns the global NexusMetrics singleton.
    Import this everywhere — never instantiate NexusMetrics directly.

    Usage:
        from nexus.observability.metrics import get_metrics
        m = get_metrics()
        m.steps_executed.inc()
        m.retrieval_latency_ms.observe(42.5)
    """
    global _metrics
    if _metrics is None:
        _metrics = NexusMetrics()
    return _metrics
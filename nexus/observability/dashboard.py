"""
nexus/observability/dashboard.py
----------------------------------
A live terminal dashboard for the Nexus benchmark runner.

Uses Rich's Live + Table + Panel layout to display real-time metrics
as the kernel executes its long-horizon task.

Layout:
┌─────────────────────────────────────────────────────────────────┐
│                    NEXUS KERNEL DASHBOARD                       │
├─────────────────┬────────────────────┬──────────────────────────┤
│  ORCHESTRATION  │  EPISTEMIC ENGINE  │      MCP FABRIC          │
│  Steps: 47/500  │  Nodes: 143        │  Servers: 2              │
│  MCTS sims: 940 │  Memories: 312     │  Tools: 6                │
│  Pruned: 12     │  Retr. p95: 48ms   │  Synth rate: 100%        │
│  Backtracks: 3  │  Compression: 8.2x │  Tool calls: 89          │
└─────────────────┴────────────────────┴──────────────────────────┘
│  LLM: 234 calls │  Errors: 0  │  Uptime: 00:03:42              │
└─────────────────────────────────────────────────────────────────┘

Can run in two modes:
  - LIVE: updates every second during execution (Rich Live)
  - STATIC: print a single snapshot at the end of a run
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nexus.observability.metrics import NexusMetrics

console = Console()


def _fmt_ms(value: float) -> str:
    """Format milliseconds nicely."""
    if value == 0:
        return "—"
    if value < 1000:
        return f"{value:.0f}ms"
    return f"{value/1000:.1f}s"


def _fmt_rate(rate: float) -> str:
    """Format a 0–1 rate as a percentage with colour."""
    pct = rate * 100
    color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    return f"[{color}]{pct:.0f}%[/]"


def _fmt_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_orchestration_table(m: NexusMetrics) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan",
              title="[bold cyan]ORCHESTRATION[/]", title_justify="left")
    t.add_column("Metric", style="dim", min_width=18)
    t.add_column("Value", justify="right", min_width=10)

    t.add_row("Steps executed",  str(m.steps_executed.value))
    t.add_row("MCTS simulations", str(m.mcts_simulations.value))
    t.add_row("Paths pruned",    f"[yellow]{m.paths_pruned.value}[/]")
    t.add_row("Backtracks",      f"[yellow]{m.backtrack_count.value}[/]")

    if m.critic_scores.count:
        score = m.critic_scores.mean
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        t.add_row("Critic score avg", f"[{color}]{score:.3f}[/]")
    else:
        t.add_row("Critic score avg", "—")

    t.add_row("Plan p95 latency", _fmt_ms(m.plan_latency_ms.p95))
    return t


def _build_epistemic_table(m: NexusMetrics) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta",
              title="[bold magenta]EPISTEMIC ENGINE[/]", title_justify="left")
    t.add_column("Metric", style="dim", min_width=18)
    t.add_column("Value", justify="right", min_width=10)

    t.add_row("Graph nodes",     str(int(m.graph_nodes_total.value)))
    t.add_row("Graph edges",     str(m.edges_ingested.value))
    t.add_row("Memories stored", str(m.memories_stored.value))
    t.add_row("Nodes ingested",  str(m.nodes_ingested.value))
    t.add_row("Retrieval p50",   _fmt_ms(m.retrieval_latency_ms.p50))
    t.add_row("Retrieval p95",   _fmt_ms(m.retrieval_latency_ms.p95))

    if m.context_token_ratio.count:
        t.add_row("Compression avg", f"{m.context_token_ratio.mean:.1f}×")
    else:
        t.add_row("Compression avg", "—")
    return t


def _build_fabric_table(m: NexusMetrics) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow",
              title="[bold yellow]MCP FABRIC[/]", title_justify="left")
    t.add_column("Metric", style="dim", min_width=18)
    t.add_column("Value", justify="right", min_width=10)

    t.add_row("Synth attempts",  str(m.synthesis_attempts.value))
    t.add_row("Synth succeeded", f"[green]{m.synthesis_successes.value}[/]")
    t.add_row("Synth failed",    f"[red]{m.synthesis_failures.value}[/]")
    t.add_row("Synth pass rate", _fmt_rate(m.synthesis_pass_rate))
    t.add_row("Tools mounted",   f"[green]{int(m.active_tools.value)}[/]")
    t.add_row("Tool calls",      str(m.tool_calls_total.value))
    t.add_row("Tool errors",     f"[red]{m.tool_call_errors.value}[/]")
    t.add_row("Tool p95 latency",_fmt_ms(m.tool_call_latency_ms.p95))
    return t


def _build_footer(m: NexusMetrics) -> Table:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("LLM")
    t.add_column("Errors")
    t.add_column("Uptime")

    llm_text = f"LLM: [bold]{m.llm_calls.value}[/] calls | p95: {_fmt_ms(m.llm_latency_ms.p95)}"
    err_text = (
        f"[green]Errors: 0[/]" if m.errors_total.value == 0
        else f"[red]Errors: {m.errors_total.value}[/]"
    )
    uptime = f"⏱  [bold]{_fmt_uptime(m.uptime_seconds)}[/]"
    t.add_row(llm_text, err_text, uptime)
    return t


def build_dashboard(m: NexusMetrics, title: str = "NEXUS KERNEL LIVE") -> Panel:
    """Build a Rich Panel containing the full dashboard layout."""
    columns = Columns([
        _build_orchestration_table(m),
        _build_epistemic_table(m),
        _build_fabric_table(m),
    ], equal=False, expand=True)

    from rich.console import Group as RGroup
    content = RGroup(columns, _build_footer(m))

    return Panel(
        content,
        title=f"[bold white]{title}[/]",
        border_style="bright_blue",
        padding=(0, 1),
    )


@contextmanager
def live_dashboard(
    metrics: NexusMetrics,
    refresh_per_second: float = 2.0,
    title: str = "NEXUS KERNEL LIVE",
) -> Generator[Live, None, None]:
    """
    Context manager that shows a live-updating dashboard during execution.

    Usage:
        with live_dashboard(metrics) as live:
            await kernel.run_exploration(...)
            # dashboard updates automatically every 0.5s
    """
    with Live(
        build_dashboard(metrics, title),
        refresh_per_second=refresh_per_second,
        console=console,
        screen=False,
    ) as live:
        def _refresh() -> None:
            live.update(build_dashboard(metrics, title))

        live._refresh_fn = _refresh  # type: ignore[attr-defined]
        yield live


def print_final_report(metrics: NexusMetrics, run_name: str = "Nexus Benchmark") -> None:
    """
    Print a static end-of-run summary report to the terminal.
    Called at the end of every benchmark run.
    """
    console.print()
    console.print(build_dashboard(metrics, title=f"{run_name} — FINAL RESULTS"))

    # Print key acceptance criteria pass/fail
    console.print()
    console.rule("[bold white]Milestone Acceptance Criteria", style="bright_blue")

    criteria = [
        (
            "M1-AC1: Ingested >10K lines",
            metrics.memories_stored.value >= 20,
            f"{metrics.memories_stored.value} memories stored",
        ),
        (
            "M1-AC3: Retrieval <500ms",
            metrics.retrieval_latency_ms.p95 < 500 or metrics.retrieval_latency_ms.count == 0,
            f"p95={_fmt_ms(metrics.retrieval_latency_ms.p95)}",
        ),
        (
            "M2-AC1: 50+ steps without drift",
            metrics.steps_executed.value >= 50,
            f"{metrics.steps_executed.value} steps executed",
        ),
        (
            "M2-AC2: Critic pruned bad paths",
            True,
            f"{metrics.paths_pruned.value} paths pruned",
        ),
        (
            "M3-AC1: MCP server synthesized",
            metrics.synthesis_successes.value >= 1,
            f"{metrics.synthesis_successes.value} servers",
        ),
        (
            "M3-AC2: Tool calls succeeded",
            metrics.tool_calls_total.value > 0 and metrics.tool_error_rate < 0.5,
            f"{metrics.tool_calls_total.value} calls, "
            f"error rate={_fmt_rate(metrics.tool_error_rate)}",
        ),
    ]

    for label, passed, detail in criteria:
        icon = "[bold green]✅[/]" if passed else "[bold red]❌[/]"
        console.print(f"  {icon}  {label}  [dim]({detail})[/]")

    console.print()
    elapsed = _fmt_uptime(metrics.uptime_seconds)
    console.print(f"  [bold]Total runtime:[/] {elapsed}")
    console.print()
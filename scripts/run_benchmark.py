#!/usr/bin/env python3
"""
scripts/run_benchmark.py
-------------------------
CLI entrypoint for the Nexus end-to-end benchmark.

Usage:
    python scripts/run_benchmark.py              # 50 steps, stub synthesis
    python scripts/run_benchmark.py --steps 100
    python scripts/run_benchmark.py --use-llm   # real gemma3 (requires ollama serve)
    make benchmark
    make benchmark-llm

Exit codes:
    0 — all core acceptance criteria passed
    1 — one or more failed
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, ".")

# ── Setup logging FIRST before any nexus imports that call log.info ──────────
# BenchmarkRunner.__init__ creates MCPFabric → OllamaClient → log.info()
# If structlog isn't configured yet, add_logger_name crashes on PrintLogger.
from nexus.observability.logging import setup_logging
setup_logging()
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nexus end-to-end benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=50,
                        help="Exploration steps (default: 50)")
    parser.add_argument("--use-llm", action="store_true", default=False,
                        help="Use real gemma3 synthesis (requires ollama serve)")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"),
                        help="Directory for JSON results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Import after setup_logging so all module-level loggers are already configured
    from benchmarks.runner import BenchmarkRunner

    print(f"\n🚀 Starting Nexus Benchmark")
    print(f"   Steps:    {args.steps}")
    print(f"   LLM:      {'gemma3 (real)' if args.use_llm else 'stubs (no Ollama)'}")
    print(f"   Seed:     {args.seed}")
    print(f"   Results:  {args.results_dir}\n")

    runner = BenchmarkRunner(
        max_steps=args.steps,
        use_llm=args.use_llm,
        results_dir=args.results_dir,
        seed=args.seed,
    )
    # runner.run() returns the flat metrics snapshot dict directly:
    # results["orchestration"], results["mcp_fabric"], etc.
    results = await runner.run()

    # Acceptance criteria check against the flat snapshot
    acs_passed = all([
        results["orchestration"]["steps_executed"] >= args.steps,
        results["mcp_fabric"]["synthesis_successes"] >= 1,
        results["mcp_fabric"]["tool_calls_total"] > 0,
    ])

    return 0 if acs_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

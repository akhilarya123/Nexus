#!/usr/bin/env python3
"""
scripts/run_benchmark.py
-------------------------
CLI entrypoint for the Nexus end-to-end benchmark.

Usage:
    # Basic run (50 steps, no Ollama needed):
    python scripts/run_benchmark.py

    # Custom steps:
    python scripts/run_benchmark.py --steps 100

    # With real LLM synthesis (requires: ollama serve && ollama pull gemma3):
    python scripts/run_benchmark.py --steps 50 --use-llm

    # Via make:
    make benchmark

Exit codes:
    0 — benchmark completed and all core ACs passed
    1 — one or more acceptance criteria failed
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, ".")

from nexus.observability.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nexus end-to-end benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of exploration steps (default: 50)",
    )
    parser.add_argument(
        "--use-llm", action="store_true", default=False,
        help="Use real Ollama/gemma3 for MCP synthesis (requires ollama serve)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("benchmarks/results"),
        help="Directory to write JSON results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    setup_logging()

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
    results = await runner.run()

    # Check core acceptance criteria
    m = results["metrics"]
    acs_passed = all([
        m["orchestration"]["steps_executed"] >= args.steps,
        m["mcp_fabric"]["synthesis_successes"] >= 1,
        m["mcp_fabric"]["tool_calls_total"] > 0,
    ])

    return 0 if acs_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
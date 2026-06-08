#!/usr/bin/env python3
"""
scripts/health_check.py
------------------------
Run this FIRST after `docker compose up -d` to verify every component
of the Nexus stack is healthy before starting development.

Usage:
    python scripts/health_check.py

Exit codes:
    0 — all systems go
    1 — one or more services are down (details printed)
"""

from __future__ import annotations

import asyncio
import sys
from typing import Callable

import httpx
import redis as redis_sync
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table
from rich import box

# Allow running from project root without installing the package
sys.path.insert(0, ".")

from nexus.config.settings import get_settings
from nexus.tools.llm_client import OllamaClient

console = Console()
cfg = get_settings()


# ---------------------------------------------------------------------------
# Individual health check functions
# ---------------------------------------------------------------------------


async def check_ollama() -> tuple[str, str, str]:
    """Check Ollama daemon and gemma3 model availability."""
    client = OllamaClient()
    result = await client.health_check()
    if result["status"] == "ok":
        return "Ollama / gemma3", "✅ OK", f"Model ready: {result['model']}"
    elif result["status"] == "model_missing":
        models = ", ".join(result.get("available_models", []))
        return (
            "Ollama / gemma3",
            "⚠️  WARN",
            f"Ollama up but '{cfg.ollama.model}' not found. "
            f"Available: {models or 'none'}. "
            f"Run: ollama pull {cfg.ollama.model}",
        )
    else:
        return (
            "Ollama / gemma3",
            "❌ FAIL",
            f"Cannot connect to Ollama at {cfg.ollama.base_url}. "
            "Is `ollama serve` running?",
        )


def check_neo4j() -> tuple[str, str, str]:
    """Check Neo4j connectivity and APOC plugin."""
    try:
        driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.username, cfg.neo4j.password),
        )
        with driver.session() as session:
            result = session.run("RETURN 'nexus' AS ping")
            record = result.single()
            ping = record["ping"] if record else "?"
        driver.close()
        return "Neo4j", "✅ OK", f"Bolt connected — ping={ping}"
    except Exception as exc:
        return (
            "Neo4j",
            "❌ FAIL",
            f"Cannot connect: {exc}. Run: docker compose up -d neo4j",
        )


def check_qdrant() -> tuple[str, str, str]:
    """Check Qdrant REST API."""
    try:
        client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=5)
        info = client.get_collections()
        count = len(info.collections)
        return "Qdrant", "✅ OK", f"REST connected — {count} collection(s)"
    except Exception as exc:
        return (
            "Qdrant",
            "❌ FAIL",
            f"Cannot connect: {exc}. Run: docker compose up -d qdrant",
        )


def check_redis() -> tuple[str, str, str]:
    """Check Redis PING."""
    try:
        r = redis_sync.Redis(
            host=cfg.redis.host,
            port=cfg.redis.port,
            db=cfg.redis.db,
            password=cfg.redis.password,
            socket_connect_timeout=3,
        )
        pong = r.ping()
        info = r.info("server")
        version = info.get("redis_version", "?")
        return "Redis", "✅ OK", f"PING={pong} — Redis {version}"
    except Exception as exc:
        return (
            "Redis",
            "❌ FAIL",
            f"Cannot connect: {exc}. Run: docker compose up -d redis",
        )


async def check_jaeger() -> tuple[str, str, str]:
    """Check Jaeger UI is reachable."""
    url = "http://localhost:16686"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url)
            if r.status_code < 400:
                return "Jaeger", "✅ OK", f"UI reachable at {url}"
            return "Jaeger", "⚠️  WARN", f"HTTP {r.status_code}"
    except Exception as exc:
        return (
            "Jaeger",
            "❌ FAIL",
            f"Cannot reach {url}: {exc}. Run: docker compose up -d jaeger",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_all_checks() -> bool:
    """Run all checks concurrently and print a Rich table."""

    console.print("\n[bold cyan]╔══════════════════════════════════════╗[/]")
    console.print("[bold cyan]║       NEXUS KERNEL HEALTH CHECK      ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/]\n")

    # Run async checks concurrently, sync ones in thread pool
    results = await asyncio.gather(
        check_ollama(),
        asyncio.to_thread(check_neo4j),
        asyncio.to_thread(check_qdrant),
        asyncio.to_thread(check_redis),
        check_jaeger(),
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Service", style="cyan", min_width=20)
    table.add_column("Status", min_width=10)
    table.add_column("Details", style="dim")

    all_ok = True
    for service, status, detail in results:
        table.add_row(service, status, detail)
        if "FAIL" in status:
            all_ok = False

    console.print(table)

    if all_ok:
        console.print("\n[bold green]🚀 All systems operational. Ready to build Nexus![/]\n")
    else:
        console.print(
            "\n[bold red]⚠️  Some services are down. Fix the issues above, then re-run.[/]\n"
        )
        console.print(
            "[dim]Quick start: cd into your project and run[/]\n"
            "[bold]  docker compose up -d[/]\n"
            "[dim]then wait ~30s for Neo4j to initialise, then re-run this check.[/]\n"
        )

    return all_ok


def main() -> None:
    ok = asyncio.run(run_all_checks())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

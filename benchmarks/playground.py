"""
nexus/benchmarks/playground.py
--------------------------------
A deterministic, synthetic infrastructure environment for benchmarking.

Simulates a realistic but entirely local multi-layer system with:
  - 3 services (api-gateway, auth-service, user-service)
  - 2 databases (postgres-main, redis-cache)
  - 1 Kubernetes cluster namespace
  - Intentional fault conditions (slow query, OOM pod, connection exhaustion)
  - 5 distinct capability gaps requiring MCP tool synthesis

Used by the BenchmarkRunner to drive the full Nexus kernel through a
realistic long-horizon systems exploration task without any real infrastructure.

The environment is stateful: each "probe" changes its state slightly,
simulating a real system under load.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServiceState:
    name: str
    healthy: bool = True
    cpu_percent: float = 0.0
    memory_mb: int = 256
    request_count: int = 0
    error_count: int = 0
    dependencies: list[str] = field(default_factory=list)

    def tick(self) -> None:
        """Advance state one time step — simulates system evolution."""
        self.request_count += random.randint(1, 20)
        self.cpu_percent = min(100.0, self.cpu_percent + random.uniform(-5, 8))
        if self.cpu_percent > 85:
            self.error_count += random.randint(0, 3)
            self.healthy = self.error_count < 10


@dataclass
class DatabaseState:
    name: str
    connection_pool_size: int = 100
    active_connections: int = 0
    slow_queries: int = 0
    tables: list[str] = field(default_factory=list)
    healthy: bool = True

    def tick(self) -> None:
        self.active_connections = min(
            self.connection_pool_size,
            self.active_connections + random.randint(-5, 10)
        )
        self.slow_queries += random.randint(0, 2)
        if self.active_connections >= self.connection_pool_size:
            self.healthy = False


class SyntheticPlayground:
    """
    The synthetic target environment.

    The BenchmarkRunner queries this instead of a real system.
    All responses are deterministic-ish (seeded randomness) for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self._step = 0
        self._start_time = time.time()

        # Services
        self.services: dict[str, ServiceState] = {
            "api-gateway": ServiceState(
                "api-gateway",
                dependencies=["auth-service", "user-service"],
                cpu_percent=22.0,
            ),
            "auth-service": ServiceState(
                "auth-service",
                dependencies=["postgres-main", "redis-cache"],
                cpu_percent=35.0,
            ),
            "user-service": ServiceState(
                "user-service",
                dependencies=["postgres-main"],
                cpu_percent=18.0,
            ),
        }

        # Databases
        self.databases: dict[str, DatabaseState] = {
            "postgres-main": DatabaseState(
                "postgres-main",
                active_connections=42,
                tables=["users", "sessions", "orders", "audit_log"],
            ),
            "redis-cache": DatabaseState(
                "redis-cache",
                connection_pool_size=200,
                active_connections=15,
            ),
        }

        # Fault catalogue — deterministic faults injected at specific steps
        self._fault_schedule = {
            10: ("postgres-main", "connection_pool_near_full"),
            20: ("api-gateway",   "high_cpu"),
            30: ("redis-cache",   "cache_miss_spike"),
            40: ("auth-service",  "pod_restart"),
            50: ("postgres-main", "slow_query_spike"),
        }

        # Capability gaps — what tools the agent needs to synthesize at each step
        self._gap_schedule = {
            5:  ("postgres_gap",   "Need to query postgres-main for slow queries",  "PostgreSQL 15"),
            15: ("redis_gap",      "Need to inspect redis-cache hit rate",           "Redis 7"),
            25: ("k8s_gap",        "Need to list and describe Kubernetes pods",       "Kubernetes 1.28"),
            35: ("network_gap",    "Need to trace network routes between services",  "Linux networking"),
            45: ("log_gap",        "Need to parse and query structured application logs", "Fluentd"),
        }

    def probe(self, target: str) -> dict[str, Any]:
        """
        Simulate probing an environment component.
        Returns realistic (but fake) observability data.
        """
        self._step += 1
        self._tick_all()

        if target in self.services:
            return self._probe_service(target)
        elif target in self.databases:
            return self._probe_database(target)
        elif target == "cluster":
            return self._probe_cluster()
        elif target == "topology":
            return self._probe_topology()
        else:
            return {"error": f"Unknown target: {target}", "step": self._step}

    def get_environment_text(self) -> str:
        """
        Generate a realistic multi-line environment description.
        Used as the initial ingestion corpus for the Epistemic Engine.
        """
        lines = []
        lines.append("=== Nexus Synthetic Benchmark Environment ===")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        lines.append("")

        for name, svc in self.services.items():
            lines.append(f"[SERVICE] {name}")
            lines.append(f"  status: {'healthy' if svc.healthy else 'degraded'}")
            lines.append(f"  cpu_percent: {svc.cpu_percent:.1f}")
            lines.append(f"  memory_mb: {svc.memory_mb}")
            lines.append(f"  depends_on: {', '.join(svc.dependencies)}")
            lines.append("")

        for name, db in self.databases.items():
            lines.append(f"[DATABASE] {name}")
            lines.append(f"  connections: {db.active_connections}/{db.connection_pool_size}")
            lines.append(f"  slow_queries: {db.slow_queries}")
            lines.append(f"  tables: {', '.join(db.tables)}")
            lines.append(f"  healthy: {db.healthy}")
            lines.append("")

        # Add realistic log lines
        lines.append("[LOGS]")
        for i in range(50):
            svc = random.choice(list(self.services.keys()))
            level = random.choice(["INFO", "INFO", "INFO", "WARN", "ERROR"])
            lines.append(
                f"  [{level}] {svc}: request_{i} completed in {random.randint(5, 2000)}ms"
            )
        lines.append("")

        return "\n".join(lines)

    def get_gap_for_step(self, step: int) -> tuple[str, str, str] | None:
        """
        Returns (gap_id, gap_description, target_system) if a capability gap
        is scheduled at this step, else None.
        """
        return self._gap_schedule.get(step)

    def get_fault_for_step(self, step: int) -> tuple[str, str] | None:
        """Returns (component, fault_type) if a fault fires at this step."""
        return self._fault_schedule.get(step)

    def get_raw_data_corpus(self, num_lines: int = 10000) -> str:
        """
        Generate a large corpus of synthetic environment data (>=10K lines).
        Used to satisfy the M1-AC1 ingestion acceptance criterion.
        """
        lines = []
        services = list(self.services.keys())
        databases = list(self.databases.keys())
        all_components = services + databases

        for i in range(num_lines):
            component = all_components[i % len(all_components)]
            log_type = i % 12
            if log_type == 0:
                lines.append(f"[INFO] {component}: handling request {i}, latency={i % 800}ms")
            elif log_type == 1:
                lines.append(f"[DEBUG] db-query: SELECT * FROM table_{i % 10} WHERE id={i}")
            elif log_type == 2:
                lines.append(f"[WARN] {component}: connection pool {i % 100}/100 used")
            elif log_type == 3:
                lines.append(f"[ERROR] pod-{component}-{i % 5}: health check failed attempt {i % 3}")
            elif log_type == 4:
                lines.append(f"[INFO] cache: key='user:{i}' hit={i % 3 != 0}")
            elif log_type == 5:
                lines.append(f"[INFO] endpoint /api/v{i % 3}/resource response={200 if i % 7 else 500}")
            elif log_type == 6:
                lines.append(f"[DEBUG] function process_{i % 8}: elapsed={i * 2}ms")
            elif log_type == 7:
                lines.append(f"[INFO] auth.handler: token verified for user_{i % 200}")
            elif log_type == 8:
                lines.append(f"[WARN] table orders_{i % 5}: row_count={i * 10}, slow_index_scan")
            elif log_type == 9:
                lines.append(f"[INFO] cluster node worker-{i % 8}: cpu={i % 100}%")
            elif log_type == 10:
                lines.append(f"[DEBUG] net: bytes_sent={i * 1024} bytes_recv={i * 512}")
            else:
                lines.append(f"[ERROR] timeout: {component} no response in 5000ms")

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────

    def _tick_all(self) -> None:
        for svc in self.services.values():
            svc.tick()
        for db in self.databases.values():
            db.tick()

    def _probe_service(self, name: str) -> dict[str, Any]:
        svc = self.services[name]
        return {
            "type": "service",
            "name": name,
            "healthy": svc.healthy,
            "cpu_percent": round(svc.cpu_percent, 1),
            "memory_mb": svc.memory_mb,
            "request_count": svc.request_count,
            "error_count": svc.error_count,
            "dependencies": svc.dependencies,
            "step": self._step,
        }

    def _probe_database(self, name: str) -> dict[str, Any]:
        db = self.databases[name]
        return {
            "type": "database",
            "name": name,
            "healthy": db.healthy,
            "active_connections": db.active_connections,
            "max_connections": db.connection_pool_size,
            "utilization_pct": round(db.active_connections / db.connection_pool_size * 100, 1),
            "slow_queries": db.slow_queries,
            "tables": db.tables,
            "step": self._step,
        }

    def _probe_cluster(self) -> dict[str, Any]:
        pods = [
            {
                "name": f"{svc}-{i}",
                "status": "Running" if self.services[svc].healthy else "CrashLoopBackOff",
                "cpu": f"{self.services[svc].cpu_percent:.0f}m",
                "memory": f"{self.services[svc].memory_mb}Mi",
            }
            for svc in self.services
            for i in range(2)
        ]
        return {"type": "cluster", "namespace": "legacy-prod", "pods": pods, "step": self._step}

    def _probe_topology(self) -> dict[str, Any]:
        edges = []
        for svc_name, svc in self.services.items():
            for dep in svc.dependencies:
                edges.append({"source": svc_name, "target": dep, "type": "DEPENDS_ON"})
        return {
            "type": "topology",
            "nodes": list(self.services.keys()) + list(self.databases.keys()),
            "edges": edges,
            "step": self._step,
        }
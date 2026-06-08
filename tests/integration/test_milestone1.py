"""
tests/integration/test_milestone1.py
--------------------------------------
Integration tests for Milestone 1: Epistemic Graph RAG & Context Routing.

REQUIRES: All Docker services running (neo4j, qdrant, redis, jaeger).
Run AFTER: make up && python scripts/health_check.py

Run with:
    pytest tests/integration/test_milestone1.py -v -s

FIXES applied vs original:
  FIX 1 — Event loop: fixtures use scope="session" (via conftest.py event_loop)
           so the Neo4j driver is never asked to use a different loop.
  FIX 2 — Qdrant API: AsyncQdrantClient.search() was removed in v1.9+;
           replaced with query_points() in vector_store.py.
  FIX 3 — AC1 line count: loop now runs 834 iterations × 12 lines = 10,008 lines.
"""

import asyncio
import time

import pytest
import pytest_asyncio

from nexus.epistemic import (
    EpistemicEngine,
    EntityType,
    GraphEdge,
    GraphNode,
    RawEnvironmentData,
    RelationType,
)


# ─────────────────────────────────────────────────────────────
# Fixtures  (scope="session" — share one engine + one event loop)
# ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def engine():
    """
    Session-scoped engine — created ONCE for the entire test run.

    FIX 1: scope="session" combined with the session-scoped event_loop
    in conftest.py means the Neo4j AsyncDriver is always used on the
    SAME event loop that created it → no more "Future attached to a
    different loop" errors.

    compress=False skips the LLM call so tests run without Ollama.
    """
    e = await EpistemicEngine.create(compress=False)
    yield e
    # Teardown — clear all test data after the full suite finishes
    try:
        await e.clear_all()
    except Exception:
        pass   # Don't let cleanup errors mask real test failures


@pytest_asyncio.fixture(scope="session")
async def seeded_engine(engine: EpistemicEngine):
    """
    Engine pre-loaded with a small synthetic infrastructure dataset.
    Seeded ONCE for the session; all read-only tests share this state.
    """
    nodes = [
        GraphNode(type=EntityType.SERVICE,   name="api-gateway",    namespace="k8s"),
        GraphNode(type=EntityType.SERVICE,   name="auth-service",   namespace="k8s"),
        GraphNode(type=EntityType.SERVICE,   name="user-service",   namespace="k8s"),
        GraphNode(type=EntityType.DATABASE,  name="postgres-main",  namespace="k8s"),
        GraphNode(type=EntityType.DATABASE,  name="redis-cache",    namespace="k8s"),
        GraphNode(type=EntityType.ENDPOINT,  name="/api/v1/login",  namespace="k8s"),
        GraphNode(type=EntityType.TABLE,     name="users",          namespace="k8s"),
        GraphNode(type=EntityType.TABLE,     name="sessions",       namespace="k8s"),
        GraphNode(type=EntityType.MODULE,    name="auth.py",        namespace="python"),
        GraphNode(type=EntityType.FUNCTION,  name="verify_token",   namespace="python"),
    ]
    saved_nodes = []
    for node in nodes:
        saved = await engine.add_node(node)
        saved_nodes.append(saved)

    name_to_id = {n.name: n.id for n in saved_nodes}

    edges_spec = [
        ("api-gateway",  "auth-service",  RelationType.CALLS),
        ("api-gateway",  "user-service",  RelationType.CALLS),
        ("auth-service", "postgres-main", RelationType.DEPENDS_ON),
        ("auth-service", "redis-cache",   RelationType.DEPENDS_ON),
        ("user-service", "postgres-main", RelationType.DEPENDS_ON),
        ("auth-service", "/api/v1/login", RelationType.EXPOSES_DATA_TO),
        ("postgres-main","users",         RelationType.CONTAINS),
        ("postgres-main","sessions",      RelationType.CONTAINS),
        ("auth.py",      "verify_token",  RelationType.CONTAINS),
        ("auth-service", "auth.py",       RelationType.IMPLEMENTS),
    ]
    for src, tgt, rel in edges_spec:
        if name_to_id.get(src) and name_to_id.get(tgt):
            await engine.add_edge(GraphEdge(
                source_id=name_to_id[src],
                target_id=name_to_id[tgt],
                relation=rel,
                weight=0.9,
            ))

    memories = [
        ("api-gateway connection to auth-service timed out after 30s",         "bash_log",    1),
        ("postgres-main: FATAL connection pool exhausted, 100/100 used",       "tool_output", 5),
        ("redis-cache: cache hit rate dropped to 12%, expected 85%+",          "tool_output", 8),
        ("auth-service pod restarted 3 times in last 10 minutes",              "bash_log",    10),
        ("SELECT * FROM users WHERE id=42 returned in 2340ms (slow query)",    "tool_output", 15),
        ("verify_token raised jwt.ExpiredSignatureError for token abc123",     "tool_output", 18),
        ("user-service dependency on postgres-main connection pool healthy",   "tool_output", 20),
        ("kubernetes pod api-gateway-7d9f crashed with OOMKilled",             "bash_log",    22),
    ]
    for content, source, step in memories:
        await engine.remember(content=content, source=source, step=step)

    return engine, name_to_id


# ─────────────────────────────────────────────────────────────
# Graph Store Tests
# ─────────────────────────────────────────────────────────────

class TestGraphStore:

    @pytest.mark.asyncio
    async def test_upsert_and_retrieve_node(self, engine):
        node = GraphNode(
            type=EntityType.SERVICE,
            name="test-service-unique-xyz",
            namespace="test",
        )
        saved = await engine.add_node(node)
        assert saved.id

        fetched = await engine.get_node(saved.id)
        assert fetched is not None
        assert fetched.name == "test-service-unique-xyz"
        assert fetched.type == EntityType.SERVICE

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(self, engine):
        node = GraphNode(
            type=EntityType.DATABASE,
            name="idempotency-test-db",
            namespace="test",
        )
        first = await engine.add_node(node)
        second = await engine.add_node(node)
        assert first.id == second.id

    @pytest.mark.asyncio
    async def test_find_nodes_by_name(self, seeded_engine):
        engine, _ = seeded_engine
        results = await engine.find_nodes(name="postgres")
        assert len(results) >= 1
        assert any("postgres" in n.name for n in results)

    @pytest.mark.asyncio
    async def test_find_nodes_by_type(self, seeded_engine):
        engine, _ = seeded_engine
        results = await engine.find_nodes(entity_type=EntityType.DATABASE)
        assert len(results) >= 2
        assert all(n.type == EntityType.DATABASE for n in results)

    @pytest.mark.asyncio
    async def test_find_nodes_by_namespace(self, seeded_engine):
        engine, _ = seeded_engine
        results = await engine.find_nodes(namespace="k8s")
        assert len(results) >= 5

    @pytest.mark.asyncio
    async def test_graph_stats(self, seeded_engine):
        engine, _ = seeded_engine
        status = await engine.status()
        assert status["graph"]["total_nodes"] >= 10
        assert status["graph"]["total_edges"] >= 8

    @pytest.mark.asyncio
    async def test_multi_hop_subgraph(self, seeded_engine):
        engine, name_to_id = seeded_engine
        auth_id = name_to_id["auth-service"]

        from nexus.epistemic.graph_store import get_graph_store
        store = await get_graph_store()
        nodes, edges = await store.get_subgraph(auth_id, depth=2)

        node_names = {n.name for n in nodes}
        assert "auth-service" in node_names
        assert "postgres-main" in node_names
        assert len(nodes) >= 3


# ─────────────────────────────────────────────────────────────
# Vector Store Tests
# ─────────────────────────────────────────────────────────────

class TestVectorStore:

    @pytest.mark.asyncio
    async def test_store_and_search_memory(self, engine):
        mem_id = await engine.remember(
            content="critical error in payment-service: stripe API timeout",
            source="tool_output",
            agent="test_agent",
            step=99,
        )
        assert mem_id

        from nexus.epistemic.vector_store import get_vector_store
        vs = await get_vector_store()
        ctx = await vs.search("payment service stripe timeout error", top_k=5)
        contents = [m.content for m in ctx.memories]
        assert any("stripe" in c.lower() or "payment" in c.lower() for c in contents)

    @pytest.mark.asyncio
    async def test_semantic_search_relevance(self, seeded_engine):
        engine, _ = seeded_engine
        from nexus.epistemic.vector_store import get_vector_store
        vs = await get_vector_store()

        ctx = await vs.search("postgres database connection problem", top_k=5)
        assert len(ctx.memories) > 0
        top_content = ctx.memories[0].content.lower()
        assert any(kw in top_content for kw in ["postgres", "connection", "pool", "users", "slow"])

    @pytest.mark.asyncio
    async def test_vector_store_stats(self, seeded_engine):
        engine, _ = seeded_engine
        status = await engine.status()
        assert status["vector"]["collection_logs"]["vectors_count"] >= 8


# ─────────────────────────────────────────────────────────────
# Context Router Tests
# ─────────────────────────────────────────────────────────────

class TestContextRouter:

    @pytest.mark.asyncio
    async def test_retrieve_returns_epistemic_context(self, seeded_engine):
        engine, _ = seeded_engine
        ctx = await engine.retrieve(
            "What does auth-service depend on?",
            compress=False,
        )
        assert ctx.query == "What does auth-service depend on?"
        assert ctx.retrieval_latency_ms > 0

    @pytest.mark.asyncio
    async def test_retrieve_finds_graph_nodes(self, seeded_engine):
        engine, _ = seeded_engine
        ctx = await engine.retrieve("postgres database tables", compress=False)
        assert len(ctx.graph_context.nodes) > 0

    @pytest.mark.asyncio
    async def test_retrieve_finds_vector_memories(self, seeded_engine):
        engine, _ = seeded_engine
        ctx = await engine.retrieve("auth service pod restart crash", compress=False)
        assert len(ctx.vector_context.memories) > 0

    @pytest.mark.asyncio
    async def test_retrieve_latency_under_500ms(self, seeded_engine):
        """AC3 pre-check: single retrieval must be under 500ms."""
        engine, _ = seeded_engine
        t0 = time.monotonic()
        await engine.retrieve("What services call postgres-main?", compress=False)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500, f"Retrieval took {elapsed_ms:.1f}ms — exceeds 500ms"

    @pytest.mark.asyncio
    async def test_to_prompt_string_is_non_empty(self, seeded_engine):
        engine, _ = seeded_engine
        ctx = await engine.retrieve("redis cache performance", compress=False)
        prompt = ctx.to_prompt_string()
        assert isinstance(prompt, str) and len(prompt) > 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_gracefully(self, engine):
        ctx = await engine.retrieve("xyzzy foobarbaz impossible_entity_abc", compress=False)
        assert ctx is not None
        assert ctx.graph_context.node_count >= 0


# ─────────────────────────────────────────────────────────────
# Milestone 1 Acceptance Criteria (combined)
# ─────────────────────────────────────────────────────────────

class TestMilestone1AcceptanceCriteria:

    @pytest.mark.asyncio
    async def test_AC1_large_ingestion(self, engine):
        """
        AC1: Ingest >10,000 lines of structured text.

        FIX 3: Original loop was 500 × 12 = 6,000 lines.
        Now 834 × 12 = 10,008 lines ✅
        """
        lines = []
        for i in range(834):   # 834 × 12 = 10,008 lines
            lines.append(f"[INFO] service-{i % 20}: handling request {i}")
            lines.append(f"[DEBUG] db-query: SELECT * FROM table_{i % 30} WHERE id={i}")
            lines.append(f"[WARN] connection pool: {i % 100}/100 connections used")
            lines.append(f"[ERROR] pod-{i % 15}: health check failed, attempt {i % 5}")
            lines.append(f"[INFO] cache: key='user:{i}' hit={i % 2 == 0}")
            lines.append(f"[INFO] endpoint /api/v{i % 3}/resource called from 10.0.{i%255}.{i%255}")
            lines.append(f"[DEBUG] function process_request_{i % 10}: elapsed={i * 3}ms")
            lines.append(f"[INFO] module auth.handler: processed token for user_{i}")
            lines.append(f"[WARN] table orders_{i % 5}: row count {i * 100}, index scan slow")
            lines.append(f"[INFO] cluster node worker-{i % 8}: CPU usage {i % 100}%")
            lines.append(f"[DEBUG] network: bytes_sent={i * 1024}, bytes_recv={i * 512}")
            lines.append(f"[ERROR] timeout: service-{i % 20} did not respond in 5000ms")

        full_log = "\n".join(lines)
        total_lines = len(full_log.splitlines())
        assert total_lines >= 10000, f"Expected >=10000 lines, got {total_lines}"

        data = RawEnvironmentData(
            source_type="bash_log",
            content=full_log,
            namespace="ac1-test",
        )

        from nexus.epistemic.ingestion import EnvironmentParser
        from nexus.epistemic.vector_store import get_vector_store
        from nexus.epistemic.models import EpisodicMemory

        parser = EnvironmentParser()
        chunks = parser.chunk(data)
        assert len(chunks) >= 100, f"Expected >=100 chunks, got {len(chunks)}"

        vs = await get_vector_store()
        memories = [
            EpisodicMemory(
                content=chunk,
                source="bash_log",
                agent="ac1_test",
                step=i,
                metadata={"namespace": "ac1-test"},
            )
            for i, chunk in enumerate(chunks[:20])
        ]
        ids = await vs.store_memories_batch(memories)
        assert len(ids) == 20

        print(f"\n✅ AC1: {total_lines} lines → {len(chunks)} chunks, 20 stored in Qdrant")

    @pytest.mark.asyncio
    async def test_AC2_multi_hop_query(self, seeded_engine):
        """
        AC2: Multi-hop graph traversal finds the correct dependency chain.
        api-gateway → auth-service → postgres-main
        """
        engine, name_to_id = seeded_engine
        from nexus.epistemic.graph_store import get_graph_store

        store = await get_graph_store()
        nodes, edges = await store.multi_hop_query(
            start_name="api-gateway",
            relation_types=[RelationType.CALLS, RelationType.DEPENDS_ON],
            depth=2,
        )
        node_names = {n.name for n in nodes}
        assert "postgres-main" in node_names
        print(f"\n✅ AC2: Multi-hop found {len(nodes)} nodes: {sorted(node_names)}")

    @pytest.mark.asyncio
    async def test_AC3_retrieval_latency(self, seeded_engine):
        """
        AC3: All 5 retrieval queries complete under 500ms each.
        """
        engine, _ = seeded_engine
        queries = [
            "postgres database connection issues",
            "auth service dependencies",
            "kubernetes pod crashes OOMKilled",
            "redis cache miss rate",
            "slow query users table",
        ]
        latencies = []
        for query in queries:
            t0 = time.monotonic()
            await engine.retrieve(query, compress=False)
            latencies.append((time.monotonic() - t0) * 1000)

        max_ms = max(latencies)
        avg_ms = sum(latencies) / len(latencies)
        print(f"\n✅ AC3: latencies={[round(l,1) for l in latencies]}ms  avg={avg_ms:.1f}  max={max_ms:.1f}")
        assert max_ms < 500, f"Max latency {max_ms:.1f}ms exceeds 500ms target"

    @pytest.mark.asyncio
    async def test_AC4_dual_layer_retrieval(self, seeded_engine):
        """
        AC4: A single retrieve() call returns BOTH graph nodes AND
        episodic memories — confirming both layers are active.
        """
        engine, _ = seeded_engine
        ctx = await engine.retrieve(
            "postgres connection pool exhausted slow query",
            compress=False,
        )
        assert ctx.graph_context.node_count > 0,  "Graph layer returned no nodes"
        assert len(ctx.vector_context.memories) > 0, "Vector layer returned no memories"

        prompt = ctx.to_prompt_string()
        assert len(prompt) > 10
        print(f"\n✅ AC4: {ctx.graph_context.node_count} graph nodes + "
              f"{len(ctx.vector_context.memories)} memories  "
              f"({ctx.retrieval_latency_ms:.0f}ms)")

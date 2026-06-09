"""
nexus/epistemic/graph_store.py
-------------------------------
Neo4j client for the Hierarchical Graph RAG layer.

Responsibilities:
  - Schema initialisation (constraints + indexes on startup)
  - CRUD for GraphNode and GraphEdge
  - Multi-hop Cypher traversal (depth-limited to prevent explosion)
  - Subgraph extraction for a given entity
  - Shortest-path queries between entities
  - Async wrapper around the sync Neo4j driver

The graph schema mirrors the Nexus architecture document:
  G = (V, E)
  V: Functions, DBTables, Endpoints, Tools, ExecutionSteps, ...
  E: DEPENDS_ON, CALLS, EXPOSES_DATA_TO, SERVICES, ...

All queries enforce cfg.neo4j.max_depth to keep traversals bounded.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from nexus.config.settings import get_settings
from nexus.epistemic.models import (
    EntityType,
    GraphEdge,
    GraphNode,
    RelationType,
)
from nexus.observability.tracing import traced

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema — run once on startup
# ---------------------------------------------------------------------------

SCHEMA_QUERIES = [
    # Uniqueness constraints (also create indexes automatically)
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name_ns IF NOT EXISTS FOR (n:Entity) REQUIRE (n.name, n.namespace) IS UNIQUE",

    # Full-text search index (for name-based lookup)
    """
    CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS
    FOR (n:Entity) ON EACH [n.name, n.embedding_text]
    """,

    # Range index for time-ordered queries
    "CREATE INDEX entity_created_at IF NOT EXISTS FOR (n:Entity) ON (n.created_at)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
    "CREATE INDEX entity_namespace IF NOT EXISTS FOR (n:Entity) ON (n.namespace)",
]


class GraphStore:
    """
    Async Neo4j client for the Nexus knowledge graph.

    Usage:
        store = GraphStore()
        await store.connect()
        await store.upsert_node(node)
        await store.upsert_edge(edge)
        subgraph = await store.get_subgraph("node-id", depth=2)
        await store.close()

    Or use as async context manager:
        async with GraphStore() as store:
            await store.upsert_node(node)
    """

    def __init__(self) -> None:
        cfg = get_settings().neo4j
        self._uri = cfg.uri
        self._auth = (cfg.username, cfg.password)
        self._database = cfg.database
        self._max_depth = cfg.max_depth
        self._max_nodes = cfg.max_nodes_per_query
        self._driver: AsyncDriver | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the driver and verify connectivity."""
        try:
            await self._dispose_driver()
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=self._auth,
                max_connection_pool_size=20,
            )
            self._loop = asyncio.get_running_loop()
            await self._driver.verify_connectivity()
            log.info("Neo4j connected", uri=self._uri)
            await self._init_schema()
        except (ServiceUnavailable, AuthError) as exc:
            log.error("Neo4j connection failed", error=str(exc))
            raise

    async def _dispose_driver(self) -> None:
        if self._driver is None:
            return

        try:
            await self._driver.close()
        except Exception:
            pass
        finally:
            self._driver = None
            self._loop = None

    async def close(self) -> None:
        if self._driver:
            await self._dispose_driver()
            log.info("Neo4j connection closed")

    async def __aenter__(self) -> "GraphStore":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    @asynccontextmanager
    async def _session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a fresh session, ensuring the driver is open."""
        current_loop = asyncio.get_running_loop()
        if self._driver is None or self._loop is not current_loop:
            await self.connect()
        async with self._driver.session(database=self._database) as session:
            yield session

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    async def _init_schema(self) -> None:
        """Run all schema queries idempotently (IF NOT EXISTS guards them)."""
        async with self._session() as session:
            for query in SCHEMA_QUERIES:
                try:
                    await session.run(query)
                except Exception as exc:
                    # Some Neo4j versions don't support all syntax — log and continue
                    log.warning("Schema query skipped", error=str(exc)[:120])
        log.info("Neo4j schema initialised")

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.graph", "upsert_node")
    async def upsert_node(self, node: GraphNode) -> GraphNode:
        """
        Insert or update a node. Uses MERGE on (name, namespace) so re-running
        ingestion doesn't create duplicates.
        """
        props = node.to_cypher_props()
        query = """
        MERGE (n:Entity {name: $name, namespace: $namespace})
        ON CREATE SET n += $props, n.id = $id
        ON MATCH  SET n += $props, n.updated_at = $updated_at
        RETURN n.id AS id
        """
        async with self._session() as session:
            result = await session.run(
                query,
                name=node.name,
                namespace=node.namespace,
                props=props,
                id=node.id,
                updated_at=node.updated_at.isoformat(),
            )
            record = await result.single()
            if record:
                node.id = record["id"]
        return node

    @traced("nexus.epistemic.graph", "get_node_by_id")
    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Fetch a single node by its UUID."""
        async with self._session() as session:
            result = await session.run(
                "MATCH (n:Entity {id: $id}) RETURN n", id=node_id
            )
            record = await result.single()
            if not record:
                return None
            return self._record_to_node(record["n"])

    @traced("nexus.epistemic.graph", "find_nodes")
    async def find_nodes(
        self,
        name: str | None = None,
        entity_type: EntityType | None = None,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Flexible node search with optional filters."""
        conditions = ["1=1"]
        params: dict[str, Any] = {"limit": limit}

        if name:
            conditions.append("toLower(n.name) CONTAINS toLower($name)")
            params["name"] = name
        if entity_type:
            conditions.append("n.type = $type")
            params["type"] = entity_type.value
        if namespace:
            conditions.append("n.namespace = $namespace")
            params["namespace"] = namespace

        where = " AND ".join(conditions)
        query = f"MATCH (n:Entity) WHERE {where} RETURN n LIMIT $limit"

        async with self._session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_node(r["n"]) for r in records]

    @traced("nexus.epistemic.graph", "delete_node")
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships."""
        async with self._session() as session:
            result = await session.run(
                "MATCH (n:Entity {id: $id}) DETACH DELETE n RETURN count(n) AS deleted",
                id=node_id,
            )
            record = await result.single()
            return bool(record and record["deleted"] > 0)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.graph", "upsert_edge")
    async def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        """
        Insert or update a directed relationship between two nodes.
        Uses MERGE so duplicate edges aren't created on re-ingestion.
        """
        props = edge.to_cypher_props()
        query = f"""
        MATCH (a:Entity {{id: $source_id}})
        MATCH (b:Entity {{id: $target_id}})
        MERGE (a)-[r:{edge.relation.value}]->(b)
        ON CREATE SET r += $props, r.id = $id
        ON MATCH  SET r.weight = $weight
        RETURN r.id AS id
        """
        async with self._session() as session:
            result = await session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                props=props,
                id=edge.id,
                weight=edge.weight,
            )
            record = await result.single()
            if record:
                edge.id = record["id"]
        return edge

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.graph", "get_subgraph")
    async def get_subgraph(
        self,
        node_id: str,
        depth: int | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        BFS expansion from a seed node up to `depth` hops.
        depth is capped at cfg.neo4j.max_depth (default 3) to prevent explosion.

        Returns:
            (nodes, edges) — all entities and relationships in the subgraph.
        """
        max_d = min(depth or self._max_depth, self._max_depth)
        query = f"""
        MATCH path = (seed:Entity {{id: $id}})-[*0..{max_d}]-(neighbor:Entity)
        WITH collect(DISTINCT path) AS paths
        UNWIND paths AS path
        UNWIND nodes(path) AS n
        WITH collect(DISTINCT n) AS all_nodes, paths
        UNWIND paths AS path
        UNWIND relationships(path) AS r
        RETURN all_nodes, collect(DISTINCT r) AS all_rels
        LIMIT $limit
        """
        async with self._session() as session:
            result = await session.run(query, id=node_id, limit=self._max_nodes)
            record = await result.single()

        if not record:
            # Seed node exists but has no relationships
            node = await self.get_node_by_id(node_id)
            return ([node] if node else [], [])

        nodes = [self._record_to_node(n) for n in record["all_nodes"]]
        edges = [self._record_to_edge(r) for r in record["all_rels"]]
        return nodes, edges

    @traced("nexus.epistemic.graph", "multi_hop_query")
    async def multi_hop_query(
        self,
        start_name: str,
        relation_types: list[RelationType] | None = None,
        target_type: EntityType | None = None,
        depth: int | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Multi-hop query from a named entity, optionally filtering by
        relationship type and/or target entity type.

        Example:
            # Find all things a Postgres service CALLS or DEPENDS_ON
            nodes, edges = await store.multi_hop_query(
                start_name="postgres-main",
                relation_types=[RelationType.CALLS, RelationType.DEPENDS_ON],
                depth=2,
            )
        """
        max_d = min(depth or self._max_depth, self._max_depth)

        # Build relationship filter
        if relation_types:
            rel_filter = "|".join(r.value for r in relation_types)
            rel_pattern = f"[r:{rel_filter}*1..{max_d}]"
        else:
            rel_pattern = f"[r*1..{max_d}]"

        # Build target type filter
        target_filter = f":Entity {{type: '{target_type.value}'}}" if target_type else ":Entity"

        query = f"""
        MATCH (start:Entity)
        WHERE toLower(start.name) = toLower($name)
        MATCH path = (start)-{rel_pattern}-(target{target_filter})
        WITH collect(DISTINCT path) AS paths
        UNWIND paths AS path
        UNWIND nodes(path) AS n
        WITH collect(DISTINCT n) AS all_nodes, paths
        UNWIND paths AS path
        UNWIND relationships(path) AS r
        RETURN all_nodes, collect(DISTINCT r) AS all_rels
        LIMIT $limit
        """
        async with self._session() as session:
            result = await session.run(
                query, name=start_name, limit=self._max_nodes
            )
            record = await result.single()

        if not record:
            return [], []

        nodes = [self._record_to_node(n) for n in record["all_nodes"]]
        edges = [self._record_to_edge(r) for r in record["all_rels"]]
        return nodes, edges

    @traced("nexus.epistemic.graph", "get_node_neighbors")
    async def get_node_neighbors(
        self, node_id: str, limit: int = 20
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Direct 1-hop neighbors only. Fast — used for quick context lookup."""
        query = """
        MATCH (n:Entity {id: $id})-[r]-(neighbor:Entity)
        RETURN collect(DISTINCT neighbor) AS nodes, collect(DISTINCT r) AS rels
        LIMIT $limit
        """
        async with self._session() as session:
            result = await session.run(query, id=node_id, limit=limit)
            record = await result.single()

        if not record:
            return [], []
        nodes = [self._record_to_node(n) for n in record["nodes"]]
        edges = [self._record_to_edge(r) for r in record["rels"]]
        return nodes, edges

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def stats(self) -> dict[str, Any]:
        """Return node and edge counts by type — useful for health checks."""
        async with self._session() as session:
            r1 = await session.run("MATCH (n:Entity) RETURN count(n) AS total_nodes")
            r2 = await session.run("MATCH ()-[r]->() RETURN count(r) AS total_edges")
            r3 = await session.run(
                "MATCH (n:Entity) RETURN n.type AS type, count(n) AS cnt ORDER BY cnt DESC"
            )
            n_rec = await r1.single()
            e_rec = await r2.single()
            type_recs = await r3.data()

        return {
            "total_nodes": n_rec["total_nodes"] if n_rec else 0,
            "total_edges": e_rec["total_edges"] if e_rec else 0,
            "nodes_by_type": {r["type"]: r["cnt"] for r in type_recs},
        }

    async def clear(self) -> None:
        """Delete ALL nodes and relationships. Use only in tests."""
        async with self._session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        log.warning("Neo4j graph cleared completely")

    # ------------------------------------------------------------------
    # Private helpers — Neo4j record → Pydantic model
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_node(record: Any) -> GraphNode:
        """Convert a raw Neo4j node record into a GraphNode."""
        props = dict(record)
        # Extract known fields, put the rest into properties
        known = {"id", "type", "name", "namespace", "embedding_text", "created_at", "updated_at"}
        extra = {
            k.removeprefix("prop_"): v
            for k, v in props.items()
            if k.startswith("prop_")
        }
        try:
            etype = EntityType(props.get("type", "UNKNOWN"))
        except ValueError:
            etype = EntityType.UNKNOWN

        return GraphNode(
            id=props.get("id", ""),
            type=etype,
            name=props.get("name", ""),
            namespace=props.get("namespace", "default"),
            embedding_text=props.get("embedding_text", ""),
            properties=extra,
        )

    @staticmethod
    def _record_to_edge(record: Any) -> GraphEdge:
        """Convert a raw Neo4j relationship record into a GraphEdge."""
        props = dict(record)
        try:
            rel = RelationType(record.type if hasattr(record, "type") else props.get("type", "RELATED_TO"))
        except (ValueError, AttributeError):
            rel = RelationType.RELATED_TO

        start_id = str(record.start_node.element_id) if hasattr(record, "start_node") else props.get("source_id", "")
        end_id = str(record.end_node.element_id) if hasattr(record, "end_node") else props.get("target_id", "")

        return GraphEdge(
            id=props.get("id", ""),
            source_id=start_id,
            target_id=end_id,
            relation=rel,
            weight=float(props.get("weight", 1.0)),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_graph_store: GraphStore | None = None


async def get_graph_store() -> GraphStore:
    """
    Returns and lazily connects the global GraphStore singleton.
    Import this in agents — don't instantiate GraphStore directly.
    """
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
        await _graph_store.connect()
    return _graph_store

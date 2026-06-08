"""
nexus/epistemic/engine.py
--------------------------
The EpistemicEngine is the single public interface for Milestone 1.
Every agent in Nexus talks to this class — never directly to the
GraphStore, VectorStore, or ContextRouter.

Public API:
    engine = await EpistemicEngine.create()

    # Ingest environment data
    stats = await engine.ingest(RawEnvironmentData(...))

    # Retrieve context for an agent prompt
    ctx = await engine.retrieve("What depends on postgres-main?")
    prompt_block = ctx.to_prompt_string()

    # Store an execution memory directly
    await engine.remember(content="Tool output: ...", source="tool_output", step=42)

    # Graph stats (for health checks / Milestone 1 verification)
    info = await engine.status()

Architecture:
    EpistemicEngine
     ├── GraphStore       (Neo4j — hierarchical entity graph)
     ├── VectorStore      (Qdrant — episodic memory)
     ├── IngestionPipeline (raw data → graph + vectors)
     └── ContextRouter    (query → EpistemicContext with LLM compression)
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from nexus.epistemic.context_router import ContextRouter, build_context_router
from nexus.epistemic.graph_store import GraphStore, get_graph_store
from nexus.epistemic.ingestion import IngestionPipeline
from nexus.epistemic.models import (
    EntityType,
    EpistemicContext,
    EpisodicMemory,
    GraphEdge,
    GraphNode,
    RawEnvironmentData,
    RelationType,
)
from nexus.epistemic.vector_store import VectorStore, get_vector_store
from nexus.observability.tracing import traced
from nexus.tools.llm_client import OllamaClient

log = structlog.get_logger(__name__)


class EpistemicEngine:
    """
    Unified facade for the entire Epistemic Engine subsystem.

    Create via:
        engine = await EpistemicEngine.create()

    All methods are async-safe. The engine holds shared singletons
    to GraphStore and VectorStore — do not create multiple instances.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        context_router: ContextRouter,
        ingestion_pipeline: IngestionPipeline,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._router = context_router
        self._ingestion = ingestion_pipeline

    # ------------------------------------------------------------------
    # Factory (always use this instead of __init__)
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        llm: OllamaClient | None = None,
        compress: bool = True,
    ) -> "EpistemicEngine":
        """
        Connect to all backends and return a ready EpistemicEngine.

        Args:
            llm:      Optional pre-built OllamaClient. If None, uses singleton.
            compress: Enable LLM context compression (disable in tests to save time).
        """
        graph_store = await get_graph_store()
        vector_store = await get_vector_store()
        context_router = await build_context_router(
            graph_store=graph_store,
            vector_store=vector_store,
            llm=llm,
            compress=compress,
        )
        ingestion_pipeline = IngestionPipeline(
            graph_store=graph_store,
            vector_store=vector_store,
            llm=llm,
        )
        engine = cls(
            graph_store=graph_store,
            vector_store=vector_store,
            context_router=context_router,
            ingestion_pipeline=ingestion_pipeline,
        )
        log.info("EpistemicEngine ready")
        return engine

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.engine", "ingest")
    async def ingest(
        self,
        data: RawEnvironmentData,
        step: int = 0,
        agent: str = "ingestion",
    ) -> dict[str, Any]:
        """
        Ingest raw environment data into the graph and vector stores.

        Args:
            data:  RawEnvironmentData with source_type, content, namespace.
            step:  Starting step index for episodic memories.
            agent: Agent name tag on stored memories.

        Returns:
            Stats dict: {"chunks": N, "nodes": N, "edges": N, "memories": N,
                         "elapsed_ms": N, "errors": N}

        Example:
            stats = await engine.ingest(RawEnvironmentData(
                source_type="bash_log",
                content=open("system_scan.log").read(),
                namespace="production-k8s",
            ))
        """
        return await self._ingestion.ingest(data, step=step, agent=agent)

    @traced("nexus.epistemic.engine", "ingest_many")
    async def ingest_many(
        self,
        dataset: list[RawEnvironmentData],
        step_offset: int = 0,
    ) -> dict[str, Any]:
        """Ingest multiple environment documents. Returns aggregated stats."""
        return await self._ingestion.ingest_many(dataset, step_offset=step_offset)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.engine", "retrieve")
    async def retrieve(
        self,
        query: str,
        top_k_vector: int = 8,
        graph_depth: int = 2,
        compress: bool | None = None,
    ) -> EpistemicContext:
        """
        The primary retrieval method — call this from planning agents.

        Runs graph + vector lookups concurrently, then compresses the
        result via LLM map-reduce into a dense context block.

        Args:
            query:         Natural language question from the planning agent.
            top_k_vector:  Number of episodic memories to retrieve.
            graph_depth:   Hop depth for graph traversal (max 3 per config).
            compress:      Override instance compression setting.

        Returns:
            EpistemicContext — call .to_prompt_string() to inject into prompts.

        Example:
            ctx = await engine.retrieve("What services depend on postgres-main?")
            system_prompt = f"You are a systems engineer.\\n\\n{ctx.to_prompt_string()}"
        """
        return await self._router.retrieve(
            query=query,
            top_k_vector=top_k_vector,
            graph_depth=graph_depth,
            compress=compress,
        )

    # ------------------------------------------------------------------
    # Direct memory write (for agents logging their own actions)
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.engine", "remember")
    async def remember(
        self,
        content: str,
        source: str = "agent_observation",
        agent: str = "unknown",
        step: int = 0,
        graph_node_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a single episodic memory directly — used by execution agents
        to log tool outputs, observations, and intermediate results.

        Returns the stored memory ID.

        Example:
            mem_id = await engine.remember(
                content="SELECT query returned 0 rows from users table",
                source="tool_output",
                agent="execution_agent",
                step=47,
            )
        """
        memory = EpisodicMemory(
            content=content,
            source=source,
            agent=agent,
            step=step,
            graph_node_ids=graph_node_ids or [],
            metadata=metadata or {},
        )
        return await self._vector.store_memory(memory)

    # ------------------------------------------------------------------
    # Direct graph write (for agents updating topology)
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.engine", "add_node")
    async def add_node(self, node: GraphNode) -> GraphNode:
        """
        Upsert a graph node directly — used by the Tool-Synthesizer agent
        to register a newly generated MCP tool into the knowledge graph.
        """
        return await self._graph.upsert_node(node)

    @traced("nexus.epistemic.engine", "add_edge")
    async def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Upsert a graph edge directly."""
        return await self._graph.upsert_edge(edge)

    @traced("nexus.epistemic.engine", "get_node")
    async def get_node(self, node_id: str) -> GraphNode | None:
        """Look up a node by ID."""
        return await self._graph.get_node_by_id(node_id)

    @traced("nexus.epistemic.engine", "find_nodes")
    async def find_nodes(
        self,
        name: str | None = None,
        entity_type: EntityType | None = None,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Search for nodes by name / type / namespace."""
        return await self._graph.find_nodes(
            name=name, entity_type=entity_type, namespace=namespace, limit=limit
        )

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        """
        Returns a full health/stats dict for all subsystems.
        Used by health checks and the Milestone 1 verification matrix.

        Returns:
            {
              "graph": {"total_nodes": N, "total_edges": N, "nodes_by_type": {...}},
              "vector": {"collection_logs": {...}, "collection_tools": {...}},
              "engine": "ok"
            }
        """
        graph_stats = await self._graph.stats()
        vector_stats = await self._vector.stats()
        return {
            "graph": graph_stats,
            "vector": vector_stats,
            "engine": "ok",
        }

    async def clear_all(self) -> None:
        """
        Wipe all data from Neo4j and Qdrant.
        ONLY call this in tests — destructive and irreversible.
        """
        await self._graph.clear()
        await self._vector.clear()
        log.warning("EpistemicEngine: all data cleared")

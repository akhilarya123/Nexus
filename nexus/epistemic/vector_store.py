"""
nexus/epistemic/vector_store.py
--------------------------------
Qdrant client for the Episodic Vector Memory layer.

Responsibilities:
  - Collection initialisation (idempotent on startup)
  - Store EpisodicMemory entries with their embeddings
  - Semantic similarity search: top-k nearest neighbours
  - Filtered search (by agent, source, step range, graph_node_id)
  - Sliding-window retrieval: most recent N entries
  - Batch upsert for ingestion pipeline efficiency

Collections managed here:
  - nexus_execution_logs  : tool calls, bash outputs, agent observations
  - nexus_tool_schemas    : synthesised MCP server schemas (added in M3)

All heavy embedding work is done by LocalEmbedder (sentence-transformers).
This class ONLY handles Qdrant I/O.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
    ScoredPoint,
)

from nexus.config.settings import get_settings
from nexus.epistemic.embeddings import get_embedder
from nexus.epistemic.models import EpisodicMemory, VectorContext
from nexus.observability.tracing import traced

log = structlog.get_logger(__name__)


class VectorStore:
    """
    Async Qdrant client for Nexus episodic memory.

    Usage:
        store = VectorStore()
        await store.connect()
        mem_id = await store.store_memory(memory)
        ctx = await store.search(query="postgres connection error", top_k=5)
        await store.close()

    Or as async context manager:
        async with VectorStore() as vs:
            await vs.store_memory(memory)
    """

    def __init__(self) -> None:
        cfg = get_settings().qdrant
        self._host = cfg.host
        self._port = cfg.port
        self._collection_logs = cfg.collection_execution_logs
        self._collection_tools = cfg.collection_tool_schemas
        self._vector_size = cfg.vector_size
        self._client: AsyncQdrantClient | None = None
        self._embedder = get_embedder()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connection and ensure collections exist."""
        self._client = AsyncQdrantClient(host=self._host, port=self._port)
        log.info("Qdrant connected", host=self._host, port=self._port)
        await self._init_collections()

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
            log.info("Qdrant connection closed")

    async def __aenter__(self) -> "VectorStore":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _require_client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("VectorStore not connected — call await store.connect() first")
        return self._client

    # ------------------------------------------------------------------
    # Collection initialisation
    # ------------------------------------------------------------------

    async def _init_collections(self) -> None:
        """Create collections if they don't already exist (idempotent)."""
        client = self._require_client()
        existing = {c.name for c in (await client.get_collections()).collections}

        for name in [self._collection_logs, self._collection_tools]:
            if name not in existing:
                await client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                log.info("Qdrant collection created", name=name)
            else:
                log.debug("Qdrant collection already exists", name=name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.vector", "store_memory")
    async def store_memory(self, memory: EpisodicMemory) -> str:
        """
        Embed and store a single EpisodicMemory.
        Returns the Qdrant point ID (same as memory.id).
        """
        client = self._require_client()
        vector = self._embedder.embed(memory.content)

        point = PointStruct(
            id=str(uuid.UUID(memory.id)),   # Qdrant requires UUID format
            vector=vector,
            payload=memory.to_qdrant_payload(),
        )
        await client.upsert(
            collection_name=self._collection_logs,
            points=[point],
        )
        log.debug(
            "Memory stored",
            id=memory.id[:8],
            source=memory.source,
            step=memory.step,
            content_len=len(memory.content),
        )
        return memory.id

    @traced("nexus.epistemic.vector", "store_memories_batch")
    async def store_memories_batch(
        self, memories: list[EpisodicMemory], batch_size: int = 64
    ) -> list[str]:
        """
        Batch upsert for the ingestion pipeline.
        Embeds all at once (faster than one-by-one) then sends to Qdrant.

        Returns list of stored memory IDs.
        """
        if not memories:
            return []

        client = self._require_client()
        texts = [m.content for m in memories]
        vectors = self._embedder.embed_batch(texts, batch_size=batch_size)

        points = [
            PointStruct(
                id=str(uuid.UUID(m.id)),
                vector=v,
                payload=m.to_qdrant_payload(),
            )
            for m, v in zip(memories, vectors)
        ]

        # Qdrant recommends batches ≤ 100 points per request
        chunk_size = 100
        for i in range(0, len(points), chunk_size):
            chunk = points[i : i + chunk_size]
            await client.upsert(
                collection_name=self._collection_logs,
                points=chunk,
            )

        log.info(
            "Batch memories stored",
            count=len(memories),
            collection=self._collection_logs,
        )
        return [m.id for m in memories]

    # ------------------------------------------------------------------
    # Read / Search operations
    # ------------------------------------------------------------------

    @traced("nexus.epistemic.vector", "search")
    async def search(
        self,
        query: str,
        top_k: int = 10,
        agent: str | None = None,
        source: str | None = None,
        min_step: int | None = None,
        max_step: int | None = None,
        graph_node_id: str | None = None,
        score_threshold: float = 0.3,
    ) -> VectorContext:
        """
        Semantic similarity search over episodic memory.

        Args:
            query:          Natural language query string.
            top_k:          Max results to return.
            agent:          Filter by which agent produced this memory.
            source:         Filter by source type ("bash_log", "tool_output", etc.).
            min_step:       Only memories from step >= min_step.
            max_step:       Only memories from step <= max_step.
            graph_node_id:  Only memories linked to this graph node ID.
            score_threshold: Minimum cosine similarity to include.

        Returns:
            VectorContext with ranked EpisodicMemory objects and scores.
        """
        client = self._require_client()
        query_vec = self._embedder.embed(query)

        # Build optional payload filter
        conditions: list[FieldCondition] = []
        if agent:
            conditions.append(FieldCondition(key="agent", match=MatchValue(value=agent)))
        if source:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
        if min_step is not None or max_step is not None:
            conditions.append(
                FieldCondition(
                    key="step",
                    range=Range(
                        gte=min_step if min_step is not None else None,
                        lte=max_step if max_step is not None else None,
                    ),
                )
            )
        if graph_node_id:
            conditions.append(
                FieldCondition(
                    key="graph_node_ids",
                    match=MatchValue(value=graph_node_id),
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results: list[ScoredPoint] = await client.search(
            collection_name=self._collection_logs,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        memories: list[EpisodicMemory] = []
        scores: list[float] = []

        for point in results:
            payload = point.payload or {}
            try:
                mem = EpisodicMemory(
                    id=payload.get("id", str(point.id)),
                    content=payload.get("content", ""),
                    source=payload.get("source", "unknown"),
                    agent=payload.get("agent", "unknown"),
                    step=payload.get("step", 0),
                    graph_node_ids=payload.get("graph_node_ids", []),
                    metadata=payload.get("metadata", {}),
                )
                memories.append(mem)
                scores.append(point.score)
            except Exception as exc:
                log.warning("Failed to deserialise memory point", error=str(exc))

        log.debug(
            "Vector search complete",
            query=query[:60],
            hits=len(memories),
            top_score=round(scores[0], 3) if scores else 0,
        )
        return VectorContext(query=query, memories=memories, scores=scores)

    @traced("nexus.epistemic.vector", "get_recent")
    async def get_recent(
        self,
        limit: int = 20,
        agent: str | None = None,
    ) -> list[EpisodicMemory]:
        """
        Retrieve the most recent N memories ordered by step (descending).
        Used by the sliding-window context builder.
        """
        client = self._require_client()

        conditions = []
        if agent:
            conditions.append(FieldCondition(key="agent", match=MatchValue(value=agent)))

        query_filter = Filter(must=conditions) if conditions else None

        # Scroll through points ordered by step field descending
        results, _ = await client.scroll(
            collection_name=self._collection_logs,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            order_by="step",
        )

        memories = []
        for point in results:
            payload = point.payload or {}
            try:
                memories.append(
                    EpisodicMemory(
                        id=payload.get("id", str(point.id)),
                        content=payload.get("content", ""),
                        source=payload.get("source", "unknown"),
                        agent=payload.get("agent", "unknown"),
                        step=payload.get("step", 0),
                        graph_node_ids=payload.get("graph_node_ids", []),
                        metadata=payload.get("metadata", {}),
                    )
                )
            except Exception as exc:
                log.warning("Failed to deserialise memory scroll point", error=str(exc))

        # Sort by step descending (most recent first)
        memories.sort(key=lambda m: m.step, reverse=True)
        return memories[:limit]

    # ------------------------------------------------------------------
    # Stats / maintenance
    # ------------------------------------------------------------------

    async def stats(self) -> dict[str, Any]:
        """Collection statistics — total vectors stored."""
        client = self._require_client()
        info_logs = await client.get_collection(self._collection_logs)
        info_tools = await client.get_collection(self._collection_tools)
        return {
            "collection_logs": {
                "name": self._collection_logs,
                "vectors_count": info_logs.vectors_count,
                "points_count": info_logs.points_count,
            },
            "collection_tools": {
                "name": self._collection_tools,
                "vectors_count": info_tools.vectors_count,
                "points_count": info_tools.points_count,
            },
        }

    async def clear(self) -> None:
        """Delete all points from all collections. Use only in tests."""
        client = self._require_client()
        for name in [self._collection_logs, self._collection_tools]:
            await client.delete_collection(name)
        await self._init_collections()
        log.warning("Qdrant collections cleared and recreated")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vector_store: VectorStore | None = None


async def get_vector_store() -> VectorStore:
    """
    Returns and lazily connects the global VectorStore singleton.
    Import this in agents — don't instantiate VectorStore directly.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        await _vector_store.connect()
    return _vector_store

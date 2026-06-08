"""
nexus/epistemic/context_router.py
----------------------------------
The Context Router is the brain of the Epistemic Engine retrieval layer.

Given a query string from a planning agent, it:
  1. Queries Neo4j for structural graph context (multi-hop subgraph)
  2. Queries Qdrant for semantically similar episodic memories
  3. Merges both into a raw EpistemicContext
  4. Runs LLM map-reduce compression to generate a dense summary
     (up to 10× token compression while retaining structural links)
  5. Returns an EpistemicContext ready to inject into agent prompts

Target: <500ms per full retrieval cycle (Milestone 1 acceptance criterion)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                   CONTEXT ROUTER                        │
    │                                                         │
    │  query ──▶ [GraphStore multi-hop] ─────┐               │
    │                                        ├──▶ [Merge]    │
    │  query ──▶ [VectorStore semantic] ─────┘       │       │
    │                                                ▼       │
    │                                   [LLM Compressor]     │
    │                                                │       │
    │                                                ▼       │
    │                                   EpistemicContext     │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import re
import time
from textwrap import dedent

import structlog

from nexus.epistemic.graph_store import GraphStore
from nexus.epistemic.models import (
    EntityType,
    EpistemicContext,
    GraphContext,
    RelationType,
    VectorContext,
)
from nexus.epistemic.vector_store import VectorStore
from nexus.observability.tracing import traced
from nexus.tools.llm_client import Message, OllamaClient

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM-based context compressor
# ---------------------------------------------------------------------------


class ContextCompressor:
    """
    Runs LLM map-reduce over raw graph + vector context to produce
    a hyper-dense, token-efficient summary for agent consumption.

    Compression ratio: up to 10× (1000 tokens of raw context → ~100 token summary)
    while retaining:
      - Entity names and types
      - Key relationships (DEPENDS_ON, CALLS, etc.)
      - Relevant failure history
      - Actionable structural facts
    """

    SYSTEM_PROMPT = dedent("""
        You are a context compression engine for an autonomous systems engineering AI.
        You receive raw infrastructure context (graph entities, relationships, and
        execution history) and compress it into a dense, actionable summary.

        Rules:
        - Output ONLY the compressed summary, no preamble
        - Preserve ALL entity names, types, and critical relationships
        - Highlight any errors, failures, or anomalies first
        - Use bullet points for entities, one line for key relationships
        - Target 150-250 words maximum
        - If there is no useful context, say "No relevant context found."
    """).strip()

    def __init__(self, llm: OllamaClient | None = None) -> None:
        self._llm = llm or OllamaClient(fast=True)  # use fast model for compression

    @traced("nexus.epistemic.router", "compress_context")
    async def compress(
        self,
        query: str,
        graph_ctx: GraphContext,
        vector_ctx: VectorContext,
    ) -> tuple[str, int]:
        """
        Compress graph + vector context into a dense summary.

        Returns:
            (summary_text, estimated_token_count)
        """
        # Build the raw context string to feed to the LLM
        raw_parts: list[str] = [f"Query: {query}\n"]

        if graph_ctx.nodes:
            nodes_text = "\n".join(
                f"  [{n.type.value}] {n.name} (ns={n.namespace})"
                + (f" — {n.properties.get('description', '')}" if n.properties.get("description") else "")
                for n in graph_ctx.nodes[:30]  # cap to avoid context overflow
            )
            raw_parts.append(f"Graph Entities ({graph_ctx.node_count}):\n{nodes_text}")

        if graph_ctx.edges:
            edges_text = "\n".join(
                f"  {e.source_id[:8]}… —[{e.relation.value}]→ {e.target_id[:8]}…"
                for e in graph_ctx.edges[:20]
            )
            raw_parts.append(f"Graph Relationships ({graph_ctx.edge_count}):\n{edges_text}")

        if vector_ctx.memories:
            mem_text = "\n".join(
                f"  [step={m.step}, {m.source}] {m.content[:300]}"
                for m in vector_ctx.memories[:8]
            )
            raw_parts.append(f"Episodic Memory (top {len(vector_ctx.memories)}):\n{mem_text}")

        raw_context = "\n\n".join(raw_parts)

        if not raw_context.strip() or (not graph_ctx.nodes and not vector_ctx.memories):
            return "No relevant context found.", 10

        prompt = f"Compress this infrastructure context into a dense summary:\n\n{raw_context}"

        try:
            response = await self._llm.chat(
                messages=[Message(role="user", content=prompt)],
                system=self.SYSTEM_PROMPT,
                max_tokens=400,  # keep compressed output tight
            )
            summary = response.content.strip()
            # Rough token estimate: ~4 chars per token
            token_estimate = len(summary) // 4
            return summary, token_estimate
        except Exception as exc:
            log.warning("Context compression failed, using raw fallback", error=str(exc))
            # Fall back to a simple truncation
            fallback = raw_context[:800]
            return fallback, len(fallback) // 4


# ---------------------------------------------------------------------------
# Main context router
# ---------------------------------------------------------------------------


class ContextRouter:
    """
    Unified retrieval interface for agent prompts.

    Usage:
        router = ContextRouter(graph_store, vector_store)
        ctx = await router.retrieve("What services depend on postgres-main?")
        # Inject ctx.to_prompt_string() into your agent's system prompt
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        llm: OllamaClient | None = None,
        compress: bool = True,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._compressor = ContextCompressor(llm=llm)
        self._compress = compress

    @traced("nexus.epistemic.router", "retrieve")
    async def retrieve(
        self,
        query: str,
        top_k_vector: int = 8,
        graph_depth: int = 2,
        entity_type_filter: EntityType | None = None,
        relation_filter: list[RelationType] | None = None,
        compress: bool | None = None,
    ) -> EpistemicContext:
        """
        Main retrieval method. Runs graph + vector lookups concurrently,
        then compresses the result.

        Args:
            query:               Natural language query from the planning agent.
            top_k_vector:        How many episodic memories to retrieve.
            graph_depth:         Max hops in the graph traversal (capped at cfg.max_depth).
            entity_type_filter:  Only traverse to nodes of this type.
            relation_filter:     Only follow these relationship types.
            compress:            Override instance-level compression setting.

        Returns:
            EpistemicContext — inject .to_prompt_string() into agent prompts.
        """
        t0 = time.monotonic()
        should_compress = compress if compress is not None else self._compress

        # --- Concurrent retrieval ---
        graph_task = asyncio.create_task(
            self._retrieve_graph(query, graph_depth, entity_type_filter, relation_filter)
        )
        vector_task = asyncio.create_task(
            self._retrieve_vector(query, top_k_vector)
        )

        graph_ctx, vector_ctx = await asyncio.gather(graph_task, vector_task)

        # --- LLM compression ---
        summary = ""
        token_estimate = 0
        if should_compress:
            summary, token_estimate = await self._compressor.compress(
                query, graph_ctx, vector_ctx
            )
        else:
            # No compression: estimate tokens from raw content
            raw_len = sum(len(n.embedding_text) for n in graph_ctx.nodes)
            raw_len += sum(len(m.content) for m in vector_ctx.memories)
            token_estimate = raw_len // 4

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        log.info(
            "Context retrieved",
            query=query[:60],
            graph_nodes=graph_ctx.node_count,
            graph_edges=graph_ctx.edge_count,
            vector_hits=len(vector_ctx.memories),
            token_estimate=token_estimate,
            elapsed_ms=elapsed_ms,
        )

        return EpistemicContext(
            query=query,
            graph_context=graph_ctx,
            vector_context=vector_ctx,
            compressed_summary=summary,
            token_estimate=token_estimate,
            retrieval_latency_ms=elapsed_ms,
        )

    @traced("nexus.epistemic.router", "retrieve_graph")
    async def _retrieve_graph(
        self,
        query: str,
        depth: int,
        entity_type: EntityType | None,
        relation_filter: list[RelationType] | None,
    ) -> GraphContext:
        """
        Find the most relevant graph entities for `query` using name matching,
        then expand their subgraph up to `depth` hops.
        """
        # Step 1: Find seed nodes by name similarity
        #   (full-text or case-insensitive CONTAINS — no embeddings in Neo4j)
        keywords = self._extract_keywords(query)
        seed_nodes = []

        for kw in keywords[:3]:  # limit to top-3 keywords
            try:
                found = await self._graph.find_nodes(name=kw, limit=3)
                seed_nodes.extend(found)
            except Exception as exc:
                log.warning("Graph keyword search failed", keyword=kw, error=str(exc))

        if not seed_nodes:
            log.debug("No seed nodes found for query", query=query[:60])
            return GraphContext(query=query)

        # Step 2: Expand subgraphs from all seed nodes concurrently
        all_nodes: dict[str, Any] = {}
        all_edges: dict[str, Any] = {}

        subgraph_tasks = [
            self._graph.get_subgraph(node.id, depth=depth)
            for node in seed_nodes[:5]  # limit to 5 seeds
        ]

        try:
            subgraph_results = await asyncio.gather(*subgraph_tasks, return_exceptions=True)
        except Exception as exc:
            log.warning("Subgraph expansion failed", error=str(exc))
            return GraphContext(query=query, nodes=seed_nodes)

        for result in subgraph_results:
            if isinstance(result, Exception):
                log.warning("Subgraph task error", error=str(result))
                continue
            nodes, edges = result
            for n in nodes:
                all_nodes[n.id] = n
            for e in edges:
                all_edges[e.id] = e

        return GraphContext(
            query=query,
            nodes=list(all_nodes.values()),
            edges=list(all_edges.values()),
            depth_reached=depth,
        )

    @traced("nexus.epistemic.router", "retrieve_vector")
    async def _retrieve_vector(self, query: str, top_k: int) -> VectorContext:
        """Semantic similarity search over episodic memory."""
        try:
            return await self._vector.search(query=query, top_k=top_k)
        except Exception as exc:
            log.warning("Vector search failed", error=str(exc))
            return VectorContext(query=query)

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """
        Simple keyword extractor — strips stop words and returns
        meaningful tokens for graph seed node lookup.
        """
        STOP_WORDS = {
            "what", "which", "who", "how", "where", "when", "is", "are",
            "all", "the", "a", "an", "in", "of", "to", "for", "and", "or",
            "that", "with", "on", "at", "by", "from", "find", "get", "show",
            "list", "tell", "me", "do", "does", "did", "have", "has", "had",
        }
        tokens = re.sub(r"[^\w\s-]", "", query.lower()).split()
        keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        # Sort by length descending (longer = more specific)
        return sorted(set(keywords), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


async def build_context_router(
    graph_store: GraphStore | None = None,
    vector_store: VectorStore | None = None,
    llm: OllamaClient | None = None,
    compress: bool = True,
) -> ContextRouter:
    """
    Convenience factory that connects stores if needed and returns a ready router.

    Usage:
        router = await build_context_router()
        ctx = await router.retrieve("postgres dependencies")
    """
    from nexus.epistemic.graph_store import get_graph_store
    from nexus.epistemic.vector_store import get_vector_store

    g = graph_store or await get_graph_store()
    v = vector_store or await get_vector_store()
    return ContextRouter(graph_store=g, vector_store=v, llm=llm, compress=compress)

"""
nexus/epistemic/models.py
--------------------------
Pydantic data models for the entire Epistemic Engine.

These are the canonical types that flow between:
  - Ingestion → Graph Store → Vector Store → Retrieval → Agents

Keep them here (not scattered across files) so imports stay clean
and agents always speak the same language.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Graph node / edge types — mirrors the Neo4j schema
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """
    Every node in the knowledge graph is one of these types.
    Matches the Nexus document's description of V (vertices).
    """
    # Infrastructure
    SERVICE = "SERVICE"           # A running service (Postgres, Redis, Kafka…)
    DATABASE = "DATABASE"         # A database instance
    TABLE = "TABLE"               # A DB table or collection
    ENDPOINT = "ENDPOINT"         # HTTP/gRPC/TCP endpoint
    CLUSTER_NODE = "CLUSTER_NODE" # A physical or virtual machine / K8s node

    # Code
    FUNCTION = "FUNCTION"         # A code function or method
    MODULE = "MODULE"             # A code module / file
    CLASS = "CLASS"               # A class definition

    # Nexus internals
    TOOL = "TOOL"                 # A synthesised MCP tool
    AGENT = "AGENT"               # A Nexus agent instance
    EXECUTION_STEP = "EXECUTION_STEP"  # A single agent action in a trajectory

    # Generic
    CONCEPT = "CONCEPT"           # Abstract concept extracted from logs
    FILE = "FILE"                 # A file on disk
    DIRECTORY = "DIRECTORY"       # A directory
    UNKNOWN = "UNKNOWN"           # Fallback — LLM could not classify


class RelationType(str, Enum):
    """
    Every directed edge in the knowledge graph is one of these types.
    Matches the Nexus document's description of E (edges).
    """
    DEPENDS_ON = "DEPENDS_ON"
    CALLS = "CALLS"
    EXPOSES_DATA_TO = "EXPOSES_DATA_TO"
    CONTAINS = "CONTAINS"          # Directory CONTAINS File
    IMPLEMENTS = "IMPLEMENTS"      # Class IMPLEMENTS Interface
    SERVICES = "SERVICES"          # Tool SERVICES infrastructure
    FAILED_AT = "FAILED_AT"        # Agent FAILED_AT Step
    PRODUCED = "PRODUCED"          # Step PRODUCED output
    RELATED_TO = "RELATED_TO"      # Generic weak relation


class GraphNode(BaseModel):
    """A single node (vertex) in the Neo4j knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EntityType
    name: str                        # Human-readable identifier
    namespace: str = "default"       # Logical grouping (e.g. "kubernetes", "postgres")
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding_text: str = ""         # Text used to generate the vector (stored for re-embedding)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_cypher_props(self) -> dict[str, Any]:
        """Flatten to a dict safe for Neo4j CREATE/MERGE properties."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "namespace": self.namespace,
            "embedding_text": self.embedding_text,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            **{f"prop_{k}": str(v) for k, v in self.properties.items()},
        }


class GraphEdge(BaseModel):
    """A directed edge between two GraphNodes."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0            # Edge confidence / strength (0–1)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_cypher_props(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            **{f"prop_{k}": str(v) for k, v in self.properties.items()},
        }


# ---------------------------------------------------------------------------
# Vector memory — Qdrant payloads
# ---------------------------------------------------------------------------


class EpisodicMemory(BaseModel):
    """
    A single episodic memory entry stored in Qdrant.
    Represents one unit of execution history (a tool call, a log line, etc.)
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str                    # Raw text content to embed + store
    source: str = "unknown"         # Where this came from: "tool_output", "bash_log", etc.
    agent: str = "unknown"          # Which agent produced this
    step: int = 0                   # Step number in the execution trajectory
    graph_node_ids: list[str] = Field(default_factory=list)  # Links to graph nodes
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Payload stored alongside the vector in Qdrant."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "agent": self.agent,
            "step": self.step,
            "graph_node_ids": self.graph_node_ids,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Ingestion inputs — what the ingestion pipeline accepts
# ---------------------------------------------------------------------------


class RawEnvironmentData(BaseModel):
    """
    Raw unstructured data from an environment scan.
    Feed this to the ingestion pipeline to get graph nodes + edges out.
    """

    source_type: str          # "bash_log" | "file_tree" | "api_schema" | "k8s_manifest" | "free_text"
    content: str              # The raw text content
    namespace: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval outputs — what agents receive from the epistemic engine
# ---------------------------------------------------------------------------


class GraphContext(BaseModel):
    """Structural context retrieved from Neo4j."""

    query: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    depth_reached: int = 0

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class VectorContext(BaseModel):
    """Episodic memory retrieved from Qdrant."""

    query: str
    memories: list[EpisodicMemory] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)  # similarity scores

    @property
    def top_memory(self) -> EpisodicMemory | None:
        return self.memories[0] if self.memories else None


class EpistemicContext(BaseModel):
    """
    The unified context package returned by the EpistemicEngine.
    This is what the Planner Agent consumes before making decisions.

    Contains:
    - graph_context: Structural topology from Neo4j
    - vector_context: Episodic history from Qdrant
    - compressed_summary: LLM-compressed dense summary (10x compression)
    - token_estimate: Rough estimate of how many tokens this uses
    """

    query: str
    graph_context: GraphContext
    vector_context: VectorContext
    compressed_summary: str = ""       # Set by ContextCompressor
    token_estimate: int = 0
    retrieval_latency_ms: float = 0.0

    def to_prompt_string(self) -> str:
        """
        Render this context as a string suitable for injection into an agent prompt.
        Uses the compressed summary if available, otherwise raw content.
        """
        parts: list[str] = []

        if self.compressed_summary:
            parts.append(f"## Compressed Context Summary\n{self.compressed_summary}")

        if self.graph_context.nodes:
            node_lines = "\n".join(
                f"  - [{n.type.value}] {n.name} (ns={n.namespace})"
                for n in self.graph_context.nodes[:20]  # cap at 20
            )
            parts.append(f"## Relevant Graph Entities ({self.graph_context.node_count} nodes)\n{node_lines}")

        if self.graph_context.edges:
            edge_lines = "\n".join(
                f"  - {e.source_id[:8]}… —[{e.relation.value}]→ {e.target_id[:8]}…"
                for e in self.graph_context.edges[:15]  # cap at 15
            )
            parts.append(f"## Graph Relationships ({self.graph_context.edge_count} edges)\n{edge_lines}")

        if self.vector_context.memories:
            mem_lines = "\n".join(
                f"  [{m.source}|step={m.step}] {m.content[:200]}"
                for m in self.vector_context.memories[:5]  # top 5
            )
            parts.append(f"## Recent Episodic Memory\n{mem_lines}")

        return "\n\n".join(parts) if parts else "No context available."

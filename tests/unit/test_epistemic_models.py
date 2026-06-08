"""
tests/unit/test_epistemic_models.py
-------------------------------------
Unit tests for Milestone 1 data models, embedder, and parser.
NO external services required — runs fully offline.

Run with: pytest tests/unit/test_epistemic_models.py -v
"""

import pytest
from uuid import UUID
from datetime import datetime

from nexus.epistemic.models import (
    EntityType,
    EpistemicContext,
    EpisodicMemory,
    GraphContext,
    GraphEdge,
    GraphNode,
    RawEnvironmentData,
    RelationType,
    VectorContext,
)
from nexus.epistemic.ingestion import EnvironmentParser, _to_graph_node, _to_graph_edge
from nexus.epistemic.ingestion import ExtractedEntity, ExtractedRelation


# ─────────────────────────────────────────────
# GraphNode
# ─────────────────────────────────────────────

class TestGraphNode:
    def test_auto_id_is_valid_uuid(self):
        node = GraphNode(type=EntityType.SERVICE, name="postgres")
        UUID(node.id)  # raises if not valid UUID

    def test_defaults(self):
        node = GraphNode(type=EntityType.DATABASE, name="users-db")
        assert node.namespace == "default"
        assert node.properties == {}
        assert isinstance(node.created_at, datetime)

    def test_cypher_props_contains_required_fields(self):
        node = GraphNode(
            type=EntityType.TABLE,
            name="orders",
            namespace="ecommerce",
            properties={"rows": "10000"},
        )
        props = node.to_cypher_props()
        assert props["type"] == "TABLE"
        assert props["name"] == "orders"
        assert props["namespace"] == "ecommerce"
        assert "prop_rows" in props

    def test_all_entity_types_are_valid(self):
        for etype in EntityType:
            node = GraphNode(type=etype, name=f"test-{etype.value}")
            assert node.type == etype


# ─────────────────────────────────────────────
# GraphEdge
# ─────────────────────────────────────────────

class TestGraphEdge:
    def test_auto_id(self):
        edge = GraphEdge(
            source_id="aaa", target_id="bbb", relation=RelationType.DEPENDS_ON
        )
        UUID(edge.id)

    def test_default_weight(self):
        edge = GraphEdge(
            source_id="x", target_id="y", relation=RelationType.CALLS
        )
        assert edge.weight == 1.0

    def test_cypher_props(self):
        edge = GraphEdge(
            source_id="x", target_id="y",
            relation=RelationType.EXPOSES_DATA_TO, weight=0.7
        )
        props = edge.to_cypher_props()
        assert props["weight"] == 0.7


# ─────────────────────────────────────────────
# EpisodicMemory
# ─────────────────────────────────────────────

class TestEpisodicMemory:
    def test_qdrant_payload_has_all_fields(self):
        mem = EpisodicMemory(
            content="SELECT * FROM users failed with timeout",
            source="tool_output",
            agent="execution_agent",
            step=42,
            graph_node_ids=["node-1", "node-2"],
        )
        payload = mem.to_qdrant_payload()
        assert payload["content"] == mem.content
        assert payload["source"] == "tool_output"
        assert payload["step"] == 42
        assert "node-1" in payload["graph_node_ids"]
        assert "timestamp" in payload

    def test_auto_uuid(self):
        mem = EpisodicMemory(content="hello")
        UUID(mem.id)


# ─────────────────────────────────────────────
# EpistemicContext.to_prompt_string()
# ─────────────────────────────────────────────

class TestEpistemicContext:
    def _make_context(self, with_data: bool = True) -> EpistemicContext:
        nodes = []
        edges = []
        memories = []
        if with_data:
            n1 = GraphNode(type=EntityType.SERVICE, name="api-gateway", namespace="k8s")
            n2 = GraphNode(type=EntityType.DATABASE, name="postgres", namespace="k8s")
            nodes = [n1, n2]
            edges = [GraphEdge(
                source_id=n1.id, target_id=n2.id, relation=RelationType.DEPENDS_ON
            )]
            memories = [EpisodicMemory(
                content="api-gateway connection to postgres timed out",
                source="bash_log",
                step=5,
            )]

        return EpistemicContext(
            query="what does api-gateway depend on?",
            graph_context=GraphContext(query="test", nodes=nodes, edges=edges),
            vector_context=VectorContext(
                query="test",
                memories=memories,
                scores=[0.92] * len(memories),
            ),
            compressed_summary="api-gateway DEPENDS_ON postgres (timeout error at step 5)",
        )

    def test_to_prompt_string_with_summary(self):
        ctx = self._make_context()
        prompt = ctx.to_prompt_string()
        assert "Compressed Context Summary" in prompt
        assert "api-gateway" in prompt

    def test_to_prompt_string_shows_nodes(self):
        ctx = self._make_context()
        prompt = ctx.to_prompt_string()
        assert "postgres" in prompt
        assert "SERVICE" in prompt or "DATABASE" in prompt

    def test_to_prompt_string_empty(self):
        ctx = self._make_context(with_data=False)
        ctx.compressed_summary = ""
        prompt = ctx.to_prompt_string()
        assert prompt == "No context available."

    def test_node_count_edge_count(self):
        ctx = self._make_context()
        assert ctx.graph_context.node_count == 2
        assert ctx.graph_context.edge_count == 1


# ─────────────────────────────────────────────
# RawEnvironmentData
# ─────────────────────────────────────────────

class TestRawEnvironmentData:
    def test_valid_source_types(self):
        for stype in ["bash_log", "file_tree", "api_schema", "k8s_manifest", "free_text"]:
            data = RawEnvironmentData(source_type=stype, content="some content")
            assert data.source_type == stype

    def test_default_namespace(self):
        data = RawEnvironmentData(source_type="bash_log", content="x")
        assert data.namespace == "default"


# ─────────────────────────────────────────────
# EnvironmentParser
# ─────────────────────────────────────────────

class TestEnvironmentParser:
    def setup_method(self):
        self.parser = EnvironmentParser()

    def test_generic_text_produces_chunks(self):
        long_text = "word " * 1000  # 5000 chars
        data = RawEnvironmentData(source_type="free_text", content=long_text)
        chunks = self.parser.chunk(data)
        assert len(chunks) >= 3

    def test_chunk_overlap_keeps_context(self):
        """Each chunk (except last) should partially overlap with the next."""
        text = "A" * 5000
        data = RawEnvironmentData(source_type="free_text", content=text)
        chunks = self.parser.chunk(data)
        # First chunk ends where second starts minus overlap
        overlap = EnvironmentParser.CHUNK_OVERLAP
        # Both chunks should share overlapping content
        assert chunks[0][-overlap:] == chunks[1][:overlap]

    def test_bash_log_splits_by_lines(self):
        lines = "\n".join([f"log line {i}: some output here" for i in range(200)])
        data = RawEnvironmentData(source_type="bash_log", content=lines)
        chunks = self.parser.chunk(data)
        assert len(chunks) > 1
        # Each chunk should have max 50 lines
        for chunk in chunks:
            assert chunk.count("\n") <= 50

    def test_yaml_splits_on_separator(self):
        yaml = "apiVersion: v1\nkind: Pod\n---\napiVersion: v1\nkind: Service"
        data = RawEnvironmentData(source_type="k8s_manifest", content=yaml)
        chunks = self.parser.chunk(data)
        assert len(chunks) == 2

    def test_empty_content_returns_no_chunks(self):
        data = RawEnvironmentData(source_type="bash_log", content="   \n  \n  ")
        chunks = self.parser.chunk(data)
        assert chunks == []

    def test_file_tree_splits_by_top_level(self):
        tree = (
            "src/\n"
            "  main.py\n"
            "  utils.py\n"
            "tests/\n"
            "  test_main.py\n"
        )
        data = RawEnvironmentData(source_type="file_tree", content=tree)
        chunks = self.parser.chunk(data)
        assert len(chunks) >= 2


# ─────────────────────────────────────────────
# Ingestion helpers
# ─────────────────────────────────────────────

class TestIngestionHelpers:
    def test_to_graph_node_valid_type(self):
        entity = ExtractedEntity(
            name="postgres", type="DATABASE",
            namespace="prod", description="Main DB"
        )
        node = _to_graph_node(entity)
        assert node.type == EntityType.DATABASE
        assert node.name == "postgres"
        assert "DATABASE" in node.embedding_text

    def test_to_graph_node_invalid_type_falls_back(self):
        entity = ExtractedEntity(name="thing", type="NOTATYPE")
        node = _to_graph_node(entity)
        assert node.type == EntityType.UNKNOWN

    def test_to_graph_edge_creates_edge(self):
        name_to_id = {"api": "id-001", "db": "id-002"}
        rel = ExtractedRelation(
            source="api", target="db", relation="DEPENDS_ON", weight=0.9
        )
        edge = _to_graph_edge(rel, name_to_id)
        assert edge is not None
        assert edge.source_id == "id-001"
        assert edge.target_id == "id-002"
        assert edge.relation == RelationType.DEPENDS_ON

    def test_to_graph_edge_missing_endpoint_returns_none(self):
        rel = ExtractedRelation(source="ghost", target="db", relation="CALLS")
        edge = _to_graph_edge(rel, {"db": "id-002"})
        assert edge is None

    def test_to_graph_edge_invalid_relation_falls_back(self):
        rel = ExtractedRelation(source="a", target="b", relation="INVENTED_RELATION")
        edge = _to_graph_edge(rel, {"a": "id-a", "b": "id-b"})
        assert edge is not None
        assert edge.relation == RelationType.RELATED_TO

    def test_weight_is_clamped(self):
        rel = ExtractedRelation(source="a", target="b", relation="CALLS", weight=999.0)
        edge = _to_graph_edge(rel, {"a": "id-a", "b": "id-b"})
        assert edge.weight <= 1.0

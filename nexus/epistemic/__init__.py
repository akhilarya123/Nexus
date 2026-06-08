"""
nexus/epistemic/__init__.py
----------------------------
Public interface for the Epistemic Engine subsystem (Milestone 1).

Agents and other modules import from here — never from the
individual submodules directly.

Usage:
    from nexus.epistemic import EpistemicEngine, RawEnvironmentData

    engine = await EpistemicEngine.create()
    await engine.ingest(RawEnvironmentData(source_type="bash_log", content="..."))
    ctx = await engine.retrieve("What services call the auth endpoint?")
    print(ctx.to_prompt_string())
"""

from nexus.epistemic.engine import EpistemicEngine
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

__all__ = [
    # Main facade
    "EpistemicEngine",
    # Data models
    "RawEnvironmentData",
    "GraphNode",
    "GraphEdge",
    "EpisodicMemory",
    "EpistemicContext",
    "GraphContext",
    "VectorContext",
    # Enums
    "EntityType",
    "RelationType",
]

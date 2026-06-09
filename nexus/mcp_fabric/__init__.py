"""
nexus/mcp_fabric/__init__.py
-----------------------------
Public interface for the Dynamic MCP Fabric (Milestone 3).

Usage:
    from nexus.mcp_fabric import MCPFabric, SynthesisRequest, ToolSpec

    fabric = MCPFabric()
    tool_names = await fabric.synthesize_and_mount(
        gap_description="Need to query an undocumented Cassandra cluster",
        target_system="Apache Cassandra 4.x",
    )
    response = await fabric.call_tool(tool_names[0], {"keyspace": "prod"})
"""

from nexus.mcp_fabric.fabric import MCPFabric
from nexus.mcp_fabric.models import (
    MCPServerSpec,
    MountedTool,
    ServerLanguage,
    ServerStatus,
    SynthesisRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolParameter,
    ToolSpec,
    ValidationResult,
)

__all__ = [
    "MCPFabric",
    "SynthesisRequest",
    "MCPServerSpec",
    "MountedTool",
    "ToolSpec",
    "ToolParameter",
    "ToolCallRequest",
    "ToolCallResponse",
    "ValidationResult",
    "ServerStatus",
    "ServerLanguage",
]

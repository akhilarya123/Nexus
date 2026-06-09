"""
nexus/mcp_fabric/models.py
---------------------------
All data types for the Dynamic MCP Fabric (Milestone 3).

The MCP (Model Context Protocol) is Anthropic's open standard for
agent-tool communication over JSON-RPC 2.0. Nexus extends this by
having the agent *write* its own MCP servers at runtime.

Key types:
  SynthesisRequest  — describes the capability gap the agent detected
  MCPServerSpec     — full specification of a generated MCP server
  ValidationResult  — output of the Critic's sandbox test run
  MountedTool       — a live, registered, callable tool in the router
  ToolCallRequest   — agent's request to invoke a mounted tool
  ToolCallResponse  — result returned to the agent
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ServerLanguage(str, Enum):
    TYPESCRIPT = "typescript"
    PYTHON     = "python"      # fallback — easier to run without Node.js


class ServerStatus(str, Enum):
    PENDING     = "pending"     # synthesis requested, not yet written
    GENERATED   = "generated"   # code written, not yet validated
    VALIDATING  = "validating"  # sandbox boot in progress
    MOUNTED     = "mounted"     # live, callable by agents
    FAILED      = "failed"      # validation or boot failed
    UNMOUNTED   = "unmounted"   # gracefully removed from router


class ValidationStage(str, Enum):
    STATIC_ANALYSIS  = "static_analysis"   # check for dangerous patterns
    SANDBOX_BOOT     = "sandbox_boot"      # can the server start?
    TOOLS_LIST       = "tools_list"        # does tools/list respond correctly?
    TOOLS_CALL       = "tools_call"        # does tools/call execute cleanly?


# ─────────────────────────────────────────────
# Synthesis request — what the Planner sends
# ─────────────────────────────────────────────

class ToolParameter(BaseModel):
    """One parameter in a tool's input schema."""
    name: str
    type: str                    # "string" | "integer" | "boolean" | "object" | "array"
    description: str = ""
    required: bool = True
    default: Any = None


class ToolSpec(BaseModel):
    """
    Describes one tool that the synthesized MCP server must expose.
    The Planner fills this in when it detects a capability gap.
    """
    name: str                           # e.g. "query_cassandra"
    description: str                    # e.g. "Execute a CQL query on Cassandra cluster"
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: str = "string"             # return type description
    example_call: dict[str, Any] = Field(default_factory=dict)


class SynthesisRequest(BaseModel):
    """
    Sent by the Planner/Synthesizer Agent to the MCP Fabric when
    it detects a tool gap.

    Example: agent hits an undocumented Postgres instance and has
    no MCP server for it → creates a SynthesisRequest.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    gap_description: str               # natural language: what capability is missing
    target_system: str = ""            # e.g. "PostgreSQL 15", "Apache Cassandra 4.x"
    tools: list[ToolSpec] = Field(default_factory=list)
    language: ServerLanguage = ServerLanguage.PYTHON
    namespace: str = "default"
    requested_by: str = "planner"
    created_at: float = Field(default_factory=time.time)


# ─────────────────────────────────────────────
# Generated server spec
# ─────────────────────────────────────────────

class MCPServerSpec(BaseModel):
    """
    Complete specification of a synthesized MCP server, including
    the generated source code and its JSON-RPC manifest.

    Written by the SynthesizerAgent, validated by the Sandbox,
    registered into the Router if it passes.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str                         # links back to SynthesisRequest
    server_name: str                        # e.g. "nexus_postgres_mcp"
    language: ServerLanguage
    source_code: str = ""                   # the generated Python/TS code
    tools: list[ToolSpec] = Field(default_factory=list)
    status: ServerStatus = ServerStatus.PENDING
    namespace: str = "default"

    # Populated after sandbox boot
    container_id: str | None = None
    port: int | None = None                 # localhost port the server listens on
    process_pid: int | None = None          # for Python subprocess servers

    created_at: float = Field(default_factory=time.time)
    mounted_at: float | None = None
    error_message: str = ""

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def to_registry_entry(self) -> dict[str, Any]:
        """Compact dict stored in the router registry."""
        return {
            "id": self.id,
            "name": self.server_name,
            "tools": self.tool_names,
            "namespace": self.namespace,
            "status": self.status.value,
            "port": self.port,
            "pid": self.process_pid,
            "mounted_at": self.mounted_at,
        }


# ─────────────────────────────────────────────
# Validation result
# ─────────────────────────────────────────────

class ValidationCheck(BaseModel):
    """Result of one specific validation stage."""
    stage: ValidationStage
    passed: bool
    message: str = ""
    latency_ms: float = 0.0


class ValidationResult(BaseModel):
    """
    Full validation report produced by the Sandbox after attempting
    to boot and test a generated MCP server.
    """
    spec_id: str
    checks: list[ValidationCheck] = Field(default_factory=list)
    overall_passed: bool = False
    failure_reason: str = ""
    tool_manifest: dict[str, Any] = Field(default_factory=dict)  # raw tools/list response
    elapsed_ms: float = 0.0

    @property
    def passed_stages(self) -> list[str]:
        return [c.stage.value for c in self.checks if c.passed]

    @property
    def failed_stage(self) -> str | None:
        for c in self.checks:
            if not c.passed:
                return c.stage.value
        return None


# ─────────────────────────────────────────────
# Mounted tool — live entry in the router
# ─────────────────────────────────────────────

class MountedTool(BaseModel):
    """
    A fully validated, live MCP tool registered in the router.
    Agents call tools through this record.
    """
    tool_name: str                          # e.g. "query_postgres"
    server_id: str                          # MCPServerSpec.id
    server_name: str
    description: str = ""
    parameters: list[ToolParameter] = Field(default_factory=list)
    namespace: str = "default"
    call_count: int = 0
    error_count: int = 0
    mounted_at: float = Field(default_factory=time.time)

    @property
    def qualified_name(self) -> str:
        """Unique name used by agents: namespace.tool_name"""
        return f"{self.namespace}.{self.tool_name}"


# ─────────────────────────────────────────────
# Tool call — agent ↔ router protocol
# ─────────────────────────────────────────────

class ToolCallRequest(BaseModel):
    """Agent's request to invoke a mounted tool."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str                  # qualified or unqualified name
    arguments: dict[str, Any] = Field(default_factory=dict)
    caller_agent: str = "executor"
    step: int = 0


class ToolCallResponse(BaseModel):
    """Result returned by the router after executing a tool call."""
    request_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: str = ""
    latency_ms: float = 0.0
    server_id: str = ""

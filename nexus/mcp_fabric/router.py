"""
nexus/mcp_fabric/router.py
---------------------------
The MCP Router is the live, hot-pluggable tool registry.

It maintains a registry of all mounted MCP servers and routes
tool call requests from agents to the correct subprocess.

Key capabilities:
  - Hot-plug: mount/unmount servers without restarting the kernel
  - Namespace isolation: tools are scoped to namespaces
  - Transparent routing: agent calls tool by name, router finds the server
  - Metrics: tracks call counts, error rates per tool
  - Thread-safe: uses asyncio.Lock for all registry mutations

This is what makes Nexus different from static tool registries:
at step N=0, the router has 0 tools. At step N=45, after the agent
synthesizes a Postgres MCP server, the router has 3 new tools —
all without any kernel restart.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog

from nexus.mcp_fabric.models import (
    MCPServerSpec,
    MountedTool,
    ServerStatus,
    ToolCallRequest,
    ToolCallResponse,
)
from nexus.mcp_fabric.sandbox import Sandbox
from nexus.observability.tracing import traced

log = structlog.get_logger(__name__)


class MCPRouter:
    """
    Hot-pluggable MCP tool registry and call router.

    Usage:
        router = MCPRouter(sandbox)
        await router.mount(spec)                     # after sandbox validation
        response = await router.call(ToolCallRequest(
            tool_name="query_postgres",
            arguments={"table": "users"},
        ))
        await router.unmount(spec.id)
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox
        # tool_name → MountedTool (qualified names: "namespace.tool_name")
        self._tools: dict[str, MountedTool] = {}
        # server_id → MCPServerSpec
        self._servers: dict[str, MCPServerSpec] = {}
        self._lock = asyncio.Lock()

    # ── Registry management ───────────────────────────────────────

    @traced("nexus.mcp_fabric.router", "mount")
    async def mount(self, spec: MCPServerSpec) -> list[str]:
        """
        Register all tools from a validated MCPServerSpec into the live registry.

        Called ONLY after sandbox validation passes.
        Returns list of qualified tool names that were mounted.

        This is the hot-plug moment: new tools become callable
        the instant this method returns.
        """
        if spec.status != ServerStatus.MOUNTED:
            raise ValueError(
                f"Cannot mount spec {spec.id}: status is {spec.status.value}, "
                f"expected 'mounted'. Run sandbox validation first."
            )

        mounted_names: list[str] = []
        async with self._lock:
            self._servers[spec.id] = spec
            for tool in spec.tools:
                mounted = MountedTool(
                    tool_name=tool.name,
                    server_id=spec.id,
                    server_name=spec.server_name,
                    description=tool.description,
                    parameters=tool.parameters,
                    namespace=spec.namespace,
                )
                # Register under both qualified and unqualified name
                self._tools[mounted.qualified_name] = mounted
                self._tools[tool.name] = mounted      # unqualified shorthand
                mounted_names.append(mounted.qualified_name)

        log.info(
            "Tools hot-plugged into router",
            server=spec.server_name,
            tools=mounted_names,
            total_tools=len(self._tools),
        )
        return mounted_names

    @traced("nexus.mcp_fabric.router", "unmount")
    async def unmount(self, server_id: str) -> list[str]:
        """
        Remove all tools from a server from the live registry.
        Also triggers sandbox teardown (kills the subprocess).

        Returns list of tool names that were removed.
        """
        async with self._lock:
            spec = self._servers.pop(server_id, None)
            if not spec:
                log.warning("Unmount: server not found", server_id=server_id)
                return []

            removed = []
            # Remove all tools belonging to this server
            to_delete = [
                name for name, tool in self._tools.items()
                if tool.server_id == server_id
            ]
            for name in to_delete:
                del self._tools[name]
                removed.append(name)

            spec.status = ServerStatus.UNMOUNTED

        # Teardown subprocess outside the lock to avoid blocking
        await self._sandbox.teardown(spec)
        log.info("Tools unmounted from router", server=spec.server_name, removed=removed)
        return removed

    # ── Tool invocation ───────────────────────────────────────────

    @traced("nexus.mcp_fabric.router", "call")
    async def call(self, request: ToolCallRequest) -> ToolCallResponse:
        """
        Route a tool call to the correct MCP server subprocess.

        Looks up the tool in the registry, finds its server process
        (managed by Sandbox), sends the JSON-RPC tools/call request,
        and returns the result.
        """
        t0 = time.monotonic()

        # Resolve tool name (try qualified first, then unqualified)
        tool = self._tools.get(request.tool_name)
        if not tool:
            return ToolCallResponse(
                request_id=request.id,
                tool_name=request.tool_name,
                success=False,
                error=f"Tool '{request.tool_name}' not found. "
                      f"Available: {self.list_tool_names()}",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )

        spec = self._servers.get(tool.server_id)
        if not spec:
            return ToolCallResponse(
                request_id=request.id,
                tool_name=request.tool_name,
                success=False,
                error=f"Server for tool '{request.tool_name}' is not running",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )

        # Get the running process from the sandbox
        proc = self._sandbox._processes.get(spec.id)
        if not proc or proc.returncode is not None:
            tool.error_count += 1
            return ToolCallResponse(
                request_id=request.id,
                tool_name=request.tool_name,
                success=False,
                error=f"Server process for '{spec.server_name}' is not running "
                      f"(returncode={proc.returncode if proc else 'None'})",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
                server_id=spec.id,
            )

        # Send tools/call via JSON-RPC
        rpc_request = json.dumps({
            "jsonrpc": "2.0",
            "id": request.id,
            "method": "tools/call",
            "params": {
                "name": tool.tool_name,
                "arguments": request.arguments,
            },
        }) + "\n"

        try:
            response = await self._sandbox._send_and_receive(proc, rpc_request)

            if "error" in response:
                tool.error_count += 1
                return ToolCallResponse(
                    request_id=request.id,
                    tool_name=request.tool_name,
                    success=False,
                    error=str(response["error"]),
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                    server_id=spec.id,
                )

            # Extract text content from MCP response
            content_blocks = response.get("result", {}).get("content", [])
            result_text = "\n".join(
                block.get("text", "") for block in content_blocks
                if block.get("type") == "text"
            )

            tool.call_count += 1
            latency = round((time.monotonic() - t0) * 1000, 1)
            log.debug(
                "Tool call succeeded",
                tool=request.tool_name,
                latency_ms=latency,
                result_len=len(result_text),
            )

            return ToolCallResponse(
                request_id=request.id,
                tool_name=request.tool_name,
                success=True,
                result=result_text,
                latency_ms=latency,
                server_id=spec.id,
            )

        except Exception as exc:
            tool.error_count += 1
            return ToolCallResponse(
                request_id=request.id,
                tool_name=request.tool_name,
                success=False,
                error=f"RPC call exception: {exc}",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
                server_id=spec.id,
            )

    # ── Query helpers ─────────────────────────────────────────────

    def list_tool_names(self, namespace: str | None = None) -> list[str]:
        """Return qualified names of all mounted tools."""
        seen: set[str] = set()
        names = []
        for name, tool in self._tools.items():
            if tool.qualified_name in seen:
                continue
            if namespace and tool.namespace != namespace:
                continue
            seen.add(tool.qualified_name)
            names.append(tool.qualified_name)
        return sorted(names)

    def get_tool(self, name: str) -> MountedTool | None:
        return self._tools.get(name)

    def list_servers(self) -> list[dict[str, Any]]:
        return [s.to_registry_entry() for s in self._servers.values()]

    @property
    def tool_count(self) -> int:
        """Number of unique mounted tools (deduplicated)."""
        return len({t.qualified_name for t in self._tools.values()})

    @property
    def server_count(self) -> int:
        return len(self._servers)

    def stats(self) -> dict[str, Any]:
        unique_tools = {t.qualified_name: t for t in self._tools.values()}
        return {
            "total_tools": len(unique_tools),
            "total_servers": self.server_count,
            "tools": [
                {
                    "name": t.qualified_name,
                    "server": t.server_name,
                    "calls": t.call_count,
                    "errors": t.error_count,
                }
                for t in unique_tools.values()
            ],
        }

"""
nexus/mcp_fabric/fabric.py
---------------------------
MCPFabric is the single public interface for Milestone 3.

It ties together Synthesizer → Sandbox → Router into one callable.
Your OrchestrationKernel calls this when the Planner signals
`needs_new_tool = True`.

Integration with your M2 kernel (kernel.py):

    from nexus.mcp_fabric import MCPFabric

    class OrchestrationKernel:
        def __init__(self):
            ...
            self.fabric = MCPFabric()   # add this

        def run_exploration(self, ...):
            ...
            # When critic detects a tool gap:
            if needs_new_tool:
                tool_names = await self.fabric.synthesize_and_mount(
                    gap_description="Need to query undocumented Postgres instance",
                    target_system="PostgreSQL 15",
                )
                state.available_tools.extend(tool_names)

            # Agent can now call the tool:
            response = await self.fabric.call_tool("query_postgres", {"table": "users"})

Full lifecycle:
    synthesize_and_mount(gap)
        → SynthesizerAgent.synthesize()   writes source code
        → Sandbox.validate()              boots subprocess, runs 4-stage checks
        → MCPRouter.mount()               hot-plugs into live registry
        → returns list of tool names      Planner adds to available_tools

    call_tool(name, args)
        → MCPRouter.call()                routes to correct subprocess
        → returns result string           Execution Agent gets output
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from nexus.mcp_fabric.models import (
    MCPServerSpec,
    ServerLanguage,
    ServerStatus,
    SynthesisRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolSpec,
)
from nexus.mcp_fabric.router import MCPRouter
from nexus.mcp_fabric.sandbox import Sandbox
from nexus.mcp_fabric.synthesizer import SynthesizerAgent
from nexus.observability.tracing import traced
from nexus.tools.llm_client import OllamaClient

log = structlog.get_logger(__name__)


class MCPFabric:
    """
    The Dynamic MCP Fabric — Nexus's self-extending tool system.

    Create once at kernel startup:
        fabric = MCPFabric()

    Then use throughout the agent's lifetime:
        # When agent hits a capability gap:
        tool_names = await fabric.synthesize_and_mount("need postgres tool")

        # When agent needs to call a tool:
        result = await fabric.call_tool("query_postgres", {"table": "users"})

        # Shutdown:
        await fabric.shutdown()
    """

    def __init__(self, llm: OllamaClient | None = None) -> None:
        self._sandbox = Sandbox()
        self._router = MCPRouter(self._sandbox)
        self._synthesizer = SynthesizerAgent(llm=llm)
        # Keeps full history of all synthesis attempts for debugging
        self._synthesis_history: list[MCPServerSpec] = []

    # ── Primary API ───────────────────────────────────────────────

    @traced("nexus.mcp_fabric", "synthesize_and_mount")
    async def synthesize_and_mount(
        self,
        gap_description: str,
        target_system: str = "",
        tools: list[ToolSpec] | None = None,
        namespace: str = "default",
        max_retries: int = 2,
    ) -> list[str]:
        """
        Full pipeline: write → validate → mount a new MCP server.

        Args:
            gap_description: Natural language description of what tool is missing.
                             e.g. "Need to query an undocumented Apache Cassandra cluster"
            target_system:   Optional hint about the target technology.
            tools:           Optional explicit ToolSpec list. If None, LLM infers them.
            namespace:       Namespace to mount tools under (e.g. "k8s", "python").
            max_retries:     How many times to retry synthesis if validation fails.
                             The failure message is fed back to the LLM for correction.

        Returns:
            List of qualified tool names now available (e.g. ["default.query_postgres"]).
            Empty list if all retries failed.
        """
        request = SynthesisRequest(
            gap_description=gap_description,
            target_system=target_system,
            tools=tools or [],
            namespace=namespace,
            language=ServerLanguage.PYTHON,
        )

        log.info(
            "synthesize_and_mount started",
            gap=gap_description[:80],
            namespace=namespace,
        )

        last_error = ""
        for attempt in range(1, max_retries + 1):
            log.info(f"Synthesis attempt {attempt}/{max_retries}")

            # If retrying, inject the previous failure into the gap description
            if last_error and attempt > 1:
                request = request.model_copy(update={
                    "gap_description": (
                        f"{gap_description}\n\n"
                        f"Previous attempt failed: {last_error}\n"
                        f"Please fix this issue in the generated code."
                    )
                })

            # 1. Synthesize
            try:
                spec = await self._synthesizer.synthesize(request)
            except Exception as exc:
                last_error = str(exc)
                log.warning("Synthesis failed", attempt=attempt, error=last_error)
                continue

            self._synthesis_history.append(spec)

            # 2. Validate in sandbox
            try:
                result = await self._sandbox.validate(spec)
            except Exception as exc:
                last_error = str(exc)
                log.warning("Sandbox error", attempt=attempt, error=last_error)
                spec.status = ServerStatus.FAILED
                spec.error_message = last_error
                continue

            if not result.overall_passed:
                last_error = result.failure_reason
                log.warning(
                    "Validation failed",
                    attempt=attempt,
                    stage=result.failed_stage,
                    reason=last_error,
                )
                continue

            # 3. Mount into router
            try:
                mounted_names = await self._router.mount(spec)
                log.info(
                    "synthesize_and_mount SUCCEEDED",
                    tools=mounted_names,
                    attempt=attempt,
                )
                return mounted_names
            except Exception as exc:
                last_error = str(exc)
                log.warning("Mount failed", attempt=attempt, error=last_error)
                continue

        log.error(
            "synthesize_and_mount FAILED after all retries",
            gap=gap_description[:80],
            last_error=last_error,
        )
        return []

    @traced("nexus.mcp_fabric", "call_tool")
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        caller_agent: str = "executor",
        step: int = 0,
    ) -> ToolCallResponse:
        """
        Call a mounted tool by name.

        Args:
            tool_name:    Qualified ("namespace.tool_name") or unqualified name.
            arguments:    Dict of argument name → value.
            caller_agent: Which agent is calling (for tracing).
            step:         Current execution step (for tracing).

        Returns:
            ToolCallResponse with .success and .result (or .error).

        Example:
            resp = await fabric.call_tool("query_postgres", {"table": "users"})
            if resp.success:
                print(resp.result)
        """
        return await self._router.call(ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments or {},
            caller_agent=caller_agent,
            step=step,
        ))

    # ── Epistemic integration ─────────────────────────────────────

    async def register_tool_in_graph(
        self,
        spec: MCPServerSpec,
        epistemic_engine: Any | None = None,
    ) -> None:
        """
        After mounting, optionally register the new tool as a node
        in the Epistemic Engine's knowledge graph.

        This lets the Planner's context retrieval find the tool
        when planning future steps.
        """
        if epistemic_engine is None:
            return
        try:
            from nexus.epistemic.models import EntityType, GraphNode, RelationType
            for tool in spec.tools:
                node = GraphNode(
                    type=EntityType.TOOL,
                    name=tool.name,
                    namespace=spec.namespace,
                    embedding_text=f"MCP tool {tool.name}: {tool.description}",
                    properties={
                        "server_name": spec.server_name,
                        "server_id": spec.id,
                        "description": tool.description,
                        "mounted_at": str(spec.mounted_at),
                    },
                )
                await epistemic_engine.add_node(node)
            log.debug("Tool registered in knowledge graph", tools=spec.tool_names)
        except Exception as exc:
            log.warning("Failed to register tool in graph", error=str(exc))

    # ── Query helpers ─────────────────────────────────────────────

    def list_tools(self, namespace: str | None = None) -> list[str]:
        """Return all currently mounted tool names."""
        return self._router.list_tool_names(namespace=namespace)

    def stats(self) -> dict[str, Any]:
        """Router + synthesis history statistics."""
        router_stats = self._router.stats()
        return {
            **router_stats,
            "synthesis_attempts": len(self._synthesis_history),
            "synthesis_succeeded": sum(
                1 for s in self._synthesis_history
                if s.status == ServerStatus.MOUNTED
            ),
            "synthesis_failed": sum(
                1 for s in self._synthesis_history
                if s.status == ServerStatus.FAILED
            ),
        }

    # ── Lifecycle ─────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully unmount all tools and kill all subprocesses."""
        for server_id in list(self._router._servers.keys()):
            try:
                await self._router.unmount(server_id)
            except Exception:
                pass
        await self._sandbox.teardown_all()
        log.info("MCPFabric shutdown complete")

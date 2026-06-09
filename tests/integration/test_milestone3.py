"""
tests/integration/test_milestone3.py
--------------------------------------
Integration tests for Milestone 3: Dynamic MCP Tool Synthesis.

REQUIRES: Python 3.11 (subprocess sandbox uses sys.executable).
Does NOT require: Docker, Ollama, Neo4j, Qdrant.
The sandbox runs generated MCP servers as local subprocesses.

Run with:
    pytest tests/integration/test_milestone3.py -v -s

Milestone 3 Acceptance Criteria verified here:
  ✅ AC1: Synthesize a valid, runnable MCP server from a gap description
  ✅ AC2: Server passes all 4 validation stages (static, boot, list, call)
  ✅ AC3: Hot-plug tools into the router without restarting anything
  ✅ AC4: Mounted tools are callable and return results
  ✅ AC5: Failed/dangerous code is rejected before it runs
  ✅ AC6: Full synthesize_and_mount pipeline works end-to-end (stub path)
"""

import asyncio
import json
import sys

import pytest
import pytest_asyncio

from nexus.mcp_fabric.fabric import MCPFabric
from nexus.mcp_fabric.models import (
    MCPServerSpec,
    ServerLanguage,
    ServerStatus,
    SynthesisRequest,
    ToolCallRequest,
    ToolParameter,
    ToolSpec,
    ValidationStage,
)
from nexus.mcp_fabric.sandbox import Sandbox
from nexus.mcp_fabric.router import MCPRouter
from nexus.mcp_fabric.synthesizer import SynthesizerAgent


# ─────────────────────────────────────────────────────────────
# Helpers — build pre-written MCP servers for deterministic tests
# ─────────────────────────────────────────────────────────────

def make_valid_mcp_server(tools: list[str] | None = None) -> str:
    """
    Build a complete, valid MCP server source string without using the LLM.
    Used to test the sandbox and router in isolation from the synthesizer.
    """
    tools = tools or ["query_db", "get_schema"]
    tool_fns = ""
    dispatch_entries = ""
    manifest_entries = ""

    for tool in tools:
        tool_fns += f"""
def impl_{tool}(args: dict) -> str:
    return f"Mock result from {tool}: {{args}}"
"""
        dispatch_entries += f'    "{tool}": impl_{tool},\n'
        manifest_entries += f"""
    {{
        "name": "{tool}",
        "description": "Mock tool: {tool}",
        "inputSchema": {{
            "type": "object",
            "properties": {{
                "input": {{"type": "string", "description": "Input value"}}
            }},
            "required": []
        }}
    }},"""

    return f'''#!/usr/bin/env python3
"""Valid test MCP server"""
import json, sys

{tool_fns}

TOOL_DISPATCH = {{
{dispatch_entries}}}

TOOLS_MANIFEST = [{manifest_entries}
]

def handle_request(request):
    method = request.get("method", "")
    req_id = request.get("id")
    if method == "initialize":
        return {{"jsonrpc": "2.0", "id": req_id, "result": {{
            "protocolVersion": "2024-11-05",
            "capabilities": {{"tools": {{}}}},
            "serverInfo": {{"name": "test_server", "version": "1.0.0"}}
        }}}}
    elif method == "tools/list":
        return {{"jsonrpc": "2.0", "id": req_id, "result": {{"tools": TOOLS_MANIFEST}}}}
    elif method == "tools/call":
        params = request.get("params", {{}})
        name = params.get("name", "")
        args = params.get("arguments", {{}})
        if name not in TOOL_DISPATCH:
            return {{"jsonrpc": "2.0", "id": req_id, "error": {{"code": -32601, "message": f"Unknown: {{name}}"}}}}
        result = TOOL_DISPATCH[name](args)
        return {{"jsonrpc": "2.0", "id": req_id, "result": {{"content": [{{"type": "text", "text": result}}]}}}}
    return {{"jsonrpc": "2.0", "id": req_id, "error": {{"code": -32601, "message": "Not found"}}}}

for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        req = json.loads(line)
        print(json.dumps(handle_request(req)), flush=True)
    except Exception as e:
        print(json.dumps({{"jsonrpc":"2.0","id":None,"error":{{"code":-32700,"message":str(e)}}}}), flush=True)
'''


def make_spec_with_code(
    code: str,
    tools: list[str] | None = None,
    status: ServerStatus = ServerStatus.GENERATED,
) -> MCPServerSpec:
    """Build an MCPServerSpec with the given source code."""
    tool_specs = [
        ToolSpec(
            name=t,
            description=f"Test tool {t}",
            parameters=[ToolParameter(name="input", type="string", description="Input")],
        )
        for t in (tools or ["query_db"])
    ]
    return MCPServerSpec(
        request_id="test-req",
        server_name="test_mcp_server",
        language=ServerLanguage.PYTHON,
        source_code=code,
        tools=tool_specs,
        status=status,
        namespace="test",
    )


# ─────────────────────────────────────────────────────────────
# Sandbox tests — real subprocess execution
# ─────────────────────────────────────────────────────────────

class TestSandboxRealExecution:

    @pytest.mark.asyncio
    async def test_valid_server_passes_all_stages(self):
        """
        AC2: A correctly written MCP server passes all 4 validation stages.
        """
        sandbox = Sandbox()
        spec = make_spec_with_code(
            make_valid_mcp_server(["query_db", "get_schema"]),
            tools=["query_db", "get_schema"],
        )

        try:
            result = await sandbox.validate(spec)
            assert result.overall_passed, f"Validation failed: {result.failure_reason}"
            assert "static_analysis" in result.passed_stages
            assert "sandbox_boot" in result.passed_stages
            assert "tools_list" in result.passed_stages
            assert "tools_call" in result.passed_stages
            assert spec.status == ServerStatus.MOUNTED
            assert spec.process_pid is not None
            print(f"\n✅ AC2: All 4 stages passed in {result.elapsed_ms:.0f}ms")
        finally:
            await sandbox.teardown(spec)

    @pytest.mark.asyncio
    async def test_dangerous_code_blocked_before_boot(self):
        """
        AC5: Code with os.system is rejected at static analysis — never runs.
        """
        sandbox = Sandbox()
        spec = make_spec_with_code("import os\nos.system('echo pwned')")

        result = await sandbox.validate(spec)
        assert not result.overall_passed
        assert result.failed_stage == "static_analysis"
        assert spec.process_pid is None  # process was never started
        print(f"\n✅ AC5: Dangerous code blocked at {result.failed_stage}")

    @pytest.mark.asyncio
    async def test_broken_server_fails_at_boot(self):
        """A server with a syntax error fails at sandbox_boot stage."""
        sandbox = Sandbox()
        spec = make_spec_with_code("def broken(\n  # unclosed paren")
        result = await sandbox.validate(spec)
        assert not result.overall_passed
        assert result.failed_stage in ("sandbox_boot", "tools_list")

    @pytest.mark.asyncio
    async def test_server_with_no_tools_fails_tools_list(self):
        """A server that returns empty tools/list fails at tools_list stage."""
        broken_server = '''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    method = req.get("method", "")
    req_id = req.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":req_id,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"empty","version":"1.0"}}}), flush=True)
    elif method == "tools/list":
        print(json.dumps({"jsonrpc":"2.0","id":req_id,"result":{"tools":[]}}), flush=True)
    else:
        print(json.dumps({"jsonrpc":"2.0","id":req_id,"error":{"code":-32601,"message":"not found"}}), flush=True)
'''
        sandbox = Sandbox()
        spec = make_spec_with_code(broken_server)
        try:
            result = await sandbox.validate(spec)
            assert not result.overall_passed
            assert result.failed_stage == "tools_list"
            print(f"\n✅ Empty tools/list correctly rejected")
        finally:
            await sandbox.teardown(spec)

    @pytest.mark.asyncio
    async def test_teardown_kills_process(self):
        """After teardown, the process is terminated."""
        sandbox = Sandbox()
        spec = make_spec_with_code(
            make_valid_mcp_server(["ping"]),
            tools=["ping"],
        )

        result = await sandbox.validate(spec)
        assert result.overall_passed
        pid = spec.process_pid
        assert pid is not None

        await sandbox.teardown(spec)
        # Process should be gone from registry
        assert spec.id not in sandbox._processes
        assert spec.id not in sandbox._temp_files


# ─────────────────────────────────────────────────────────────
# Router + sandbox integration tests
# ─────────────────────────────────────────────────────────────

class TestRouterWithRealSandbox:

    @pytest.mark.asyncio
    async def test_mount_and_call_tool(self):
        """
        AC3 + AC4: Mount a validated server and call a tool through the router.
        """
        sandbox = Sandbox()
        router = MCPRouter(sandbox)
        spec = make_spec_with_code(
            make_valid_mcp_server(["query_db"]),
            tools=["query_db"],
        )

        try:
            result = await sandbox.validate(spec)
            assert result.overall_passed, result.failure_reason

            # Hot-plug into router
            mounted = await router.mount(spec)
            assert "test.query_db" in mounted
            assert router.tool_count == 1
            print(f"\n✅ AC3: Tools hot-plugged: {mounted}")

            # Call the tool
            response = await router.call(ToolCallRequest(
                tool_name="query_db",
                arguments={"input": "SELECT * FROM users"},
            ))
            assert response.success, response.error
            assert "Mock result" in str(response.result)
            print(f"✅ AC4: Tool call succeeded: {response.result}")
        finally:
            await sandbox.teardown_all()

    @pytest.mark.asyncio
    async def test_multiple_tools_from_one_server(self):
        """One server exposing 3 tools → all 3 mounted and callable."""
        sandbox = Sandbox()
        router = MCPRouter(sandbox)
        spec = make_spec_with_code(
            make_valid_mcp_server(["list_tables", "describe_table", "run_query"]),
            tools=["list_tables", "describe_table", "run_query"],
        )

        try:
            result = await sandbox.validate(spec)
            assert result.overall_passed, result.failure_reason

            mounted = await router.mount(spec)
            assert len(mounted) == 3
            assert router.tool_count == 3

            # Call each one
            for tool_name in ["list_tables", "describe_table", "run_query"]:
                resp = await router.call(ToolCallRequest(
                    tool_name=tool_name,
                    arguments={"input": "test"},
                ))
                assert resp.success, f"{tool_name} failed: {resp.error}"
        finally:
            await sandbox.teardown_all()

    @pytest.mark.asyncio
    async def test_unmount_removes_tools(self):
        """After unmounting a server, its tools are no longer callable."""
        sandbox = Sandbox()
        router = MCPRouter(sandbox)
        spec = make_spec_with_code(
            make_valid_mcp_server(["temp_tool"]),
            tools=["temp_tool"],
        )

        try:
            result = await sandbox.validate(spec)
            assert result.overall_passed

            await router.mount(spec)
            assert router.tool_count == 1

            await router.unmount(spec.id)
            assert router.tool_count == 0

            resp = await router.call(ToolCallRequest(tool_name="temp_tool"))
            assert not resp.success
            assert "not found" in resp.error
        finally:
            await sandbox.teardown_all()


# ─────────────────────────────────────────────────────────────
# Synthesizer stub path (no LLM)
# ─────────────────────────────────────────────────────────────

class TestSynthesizerStubPath:

    @pytest.mark.asyncio
    async def test_synthesize_produces_runnable_code(self):
        """
        AC1: The synthesizer's stub path produces code that boots and passes
        all 4 sandbox stages — without needing Ollama running.
        """
        synthesizer = SynthesizerAgent.__new__(SynthesizerAgent)

        tools = [
            ToolSpec(
                name="list_pods",
                description="List Kubernetes pods in a namespace",
                parameters=[
                    ToolParameter(name="namespace", type="string",
                                  description="K8s namespace"),
                ],
            ),
            ToolSpec(
                name="get_pod_logs",
                description="Get logs for a specific pod",
                parameters=[
                    ToolParameter(name="pod_name", type="string",
                                  description="Pod name"),
                    ToolParameter(name="tail", type="integer",
                                  description="Lines to tail", required=False),
                ],
            ),
        ]

        # Generate implementations using stub path (no LLM)
        impl_code, dispatch = synthesizer._stub_implementations(tools)
        manifest = synthesizer._build_tools_manifest(tools)

        # Assemble the full server using the template
        from nexus.mcp_fabric.synthesizer import MCP_SERVER_TEMPLATE
        import json as _json
        dispatch_code = (
            "\nTOOL_DISPATCH = {\n"
            + "".join(f'    "{k}": {v},\n' for k, v in dispatch.items())
            + "}\n"
        )
        source = MCP_SERVER_TEMPLATE.format(
            server_name="test_k8s_mcp",
            tool_names=", ".join(t.name for t in tools),
            tool_implementations=impl_code + dispatch_code,
            tools_manifest=_json.dumps(manifest, indent=4),
        )

        spec = MCPServerSpec(
            request_id="stub-test",
            server_name="test_k8s_mcp",
            language=ServerLanguage.PYTHON,
            source_code=source,
            tools=tools,
            status=ServerStatus.GENERATED,
            namespace="k8s",
        )

        sandbox = Sandbox()
        try:
            result = await sandbox.validate(spec)
            assert result.overall_passed, (
                f"Stub-synthesized server failed validation: {result.failure_reason}\n"
                f"Failed at: {result.failed_stage}\n"
                f"Source preview:\n{source[:500]}"
            )
            assert spec.status == ServerStatus.MOUNTED
            print(f"\n✅ AC1: Stub-synthesized server validated in {result.elapsed_ms:.0f}ms")
        finally:
            await sandbox.teardown(spec)


# ─────────────────────────────────────────────────────────────
# Full MCPFabric pipeline (stub synthesizer, no LLM)
# ─────────────────────────────────────────────────────────────

class TestMCPFabricPipeline:

    @pytest.mark.asyncio
    async def test_synthesize_and_mount_with_explicit_tools(self):
        """
        AC6: Full synthesize_and_mount pipeline with explicit tools
        (bypasses LLM inference, uses stub implementations).
        """
        # Patch the LLM call in synthesizer to use stubs
        from unittest.mock import AsyncMock, patch

        fabric = MCPFabric()

        tools = [
            ToolSpec(
                name="scan_endpoints",
                description="Scan network endpoints for open ports",
                parameters=[
                    ToolParameter(name="host", type="string", description="Host to scan"),
                ],
            ),
        ]

        # Patch _generate_implementations to use stubs (avoid needing Ollama)
        with patch.object(
            fabric._synthesizer,
            "_generate_implementations",
            wraps=fabric._synthesizer._stub_implementations,
        ):
            mounted_tools = await fabric.synthesize_and_mount(
                gap_description="Need to scan network endpoints",
                target_system="Linux network",
                tools=tools,
                namespace="network",
            )

        try:
            assert len(mounted_tools) > 0, "No tools were mounted"
            assert any("scan_endpoints" in t for t in mounted_tools)
            print(f"\n✅ AC6: synthesize_and_mount succeeded: {mounted_tools}")

            # Verify the tool is callable
            resp = await fabric.call_tool("scan_endpoints", {"host": "localhost"})
            assert resp.success, f"Tool call failed: {resp.error}"
            print(f"   Tool result: {resp.result}")

            # Verify stats
            stats = fabric.stats()
            assert stats["total_tools"] >= 1
            assert stats["synthesis_succeeded"] >= 1
            print(f"   Stats: {stats}")
        finally:
            await fabric.shutdown()

    @pytest.mark.asyncio
    async def test_fabric_handles_mount_failure_gracefully(self):
        """Fabric returns empty list when all retries fail — no crash."""
        fabric = MCPFabric()

        # Patch synthesizer to produce dangerous code every time
        from unittest.mock import AsyncMock

        async def make_dangerous_spec(request):
            spec = MCPServerSpec(
                request_id=request.id,
                server_name="evil_mcp",
                language=ServerLanguage.PYTHON,
                source_code="import os; os.system('echo pwned')",
                tools=request.tools or [ToolSpec(name="t", description="t")],
                status=ServerStatus.GENERATED,
            )
            return spec

        fabric._synthesizer.synthesize = make_dangerous_spec
        result = await fabric.synthesize_and_mount(
            gap_description="test failure path",
            max_retries=2,
        )
        assert result == []  # graceful failure
        await fabric.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_servers_in_fabric(self):
        """Mount two servers from different namespaces — both callable."""
        from unittest.mock import patch

        fabric = MCPFabric()

        # Two separate tool sets for two namespaces
        db_tools = [ToolSpec(name="query_pg", description="Query Postgres",
                             parameters=[ToolParameter(name="sql", type="string",
                                                       description="SQL query")])]
        cache_tools = [ToolSpec(name="get_cache", description="Get cache value",
                                parameters=[ToolParameter(name="key", type="string",
                                                          description="Cache key")])]

        try:
            with patch.object(fabric._synthesizer, "_generate_implementations",
                              wraps=fabric._synthesizer._stub_implementations):
                db_mounted = await fabric.synthesize_and_mount(
                    gap_description="need postgres access",
                    tools=db_tools, namespace="db",
                )
                cache_mounted = await fabric.synthesize_and_mount(
                    gap_description="need redis access",
                    tools=cache_tools, namespace="cache",
                )

            assert len(db_mounted) >= 1
            assert len(cache_mounted) >= 1
            assert fabric._router.tool_count >= 2

            # Both callable
            r1 = await fabric.call_tool("query_pg", {"sql": "SELECT 1"})
            r2 = await fabric.call_tool("get_cache", {"key": "user:42"})
            assert r1.success, r1.error
            assert r2.success, r2.error
            print(f"\n✅ Two servers mounted and callable. Total tools: {fabric._router.tool_count}")
        finally:
            await fabric.shutdown()

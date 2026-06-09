"""
tests/unit/test_mcp_models.py
------------------------------
Unit tests for Milestone 3 MCP Fabric.
NO external services, Docker, or Ollama required.

Tests cover:
  - All model types and their helpers
  - Sandbox static analysis (dangerous pattern detection)
  - Router registry operations (mount, unmount, lookup)
  - Synthesizer template assembly (stub path, no LLM)

Run with: pytest tests/unit/test_mcp_models.py -v
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    ValidationCheck,
    ValidationResult,
    ValidationStage,
)
from nexus.mcp_fabric.sandbox import DANGEROUS_PATTERNS, Sandbox
from nexus.mcp_fabric.router import MCPRouter
from nexus.mcp_fabric.synthesizer import SynthesizerAgent


# ─────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────

class TestSynthesisRequest:
    def test_auto_id(self):
        req = SynthesisRequest(gap_description="need postgres tool")
        assert req.id
        assert len(req.id) == 36  # UUID format

    def test_defaults(self):
        req = SynthesisRequest(gap_description="test")
        assert req.language == ServerLanguage.PYTHON
        assert req.namespace == "default"
        assert req.tools == []

    def test_with_tools(self):
        req = SynthesisRequest(
            gap_description="query cassandra",
            tools=[ToolSpec(name="cql_query", description="Run CQL")],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "cql_query"


class TestMCPServerSpec:
    def _make_spec(self, status=ServerStatus.MOUNTED) -> MCPServerSpec:
        return MCPServerSpec(
            request_id="req-1",
            server_name="nexus_test_mcp",
            language=ServerLanguage.PYTHON,
            source_code="print('hello')",
            tools=[
                ToolSpec(name="tool_a", description="Tool A"),
                ToolSpec(name="tool_b", description="Tool B"),
            ],
            status=status,
            namespace="test",
        )

    def test_tool_names(self):
        spec = self._make_spec()
        assert spec.tool_names == ["tool_a", "tool_b"]

    def test_to_registry_entry(self):
        spec = self._make_spec()
        entry = spec.to_registry_entry()
        assert entry["name"] == "nexus_test_mcp"
        assert "tool_a" in entry["tools"]
        assert entry["status"] == "mounted"

    def test_status_default_pending(self):
        spec = MCPServerSpec(
            request_id="r", server_name="s", language=ServerLanguage.PYTHON
        )
        assert spec.status == ServerStatus.PENDING


class TestToolSpec:
    def test_with_parameters(self):
        tool = ToolSpec(
            name="query_postgres",
            description="Run a SQL query",
            parameters=[
                ToolParameter(name="table", type="string", description="Table name"),
                ToolParameter(name="limit", type="integer", description="Row limit"),
            ],
        )
        assert len(tool.parameters) == 2
        assert tool.parameters[0].required is True

    def test_optional_parameter(self):
        param = ToolParameter(name="limit", type="integer", required=False, default=100)
        assert param.required is False
        assert param.default == 100


class TestValidationResult:
    def test_passed_stages(self):
        result = ValidationResult(
            spec_id="s",
            checks=[
                ValidationCheck(stage=ValidationStage.STATIC_ANALYSIS, passed=True),
                ValidationCheck(stage=ValidationStage.SANDBOX_BOOT, passed=True),
                ValidationCheck(stage=ValidationStage.TOOLS_LIST, passed=False,
                                message="empty tools"),
            ],
            overall_passed=False,
        )
        assert "static_analysis" in result.passed_stages
        assert "sandbox_boot" in result.passed_stages
        assert result.failed_stage == "tools_list"

    def test_all_passed(self):
        result = ValidationResult(
            spec_id="s",
            checks=[
                ValidationCheck(stage=s, passed=True)
                for s in ValidationStage
            ],
            overall_passed=True,
        )
        assert result.failed_stage is None


class TestMountedTool:
    def test_qualified_name(self):
        tool = MountedTool(
            tool_name="query_postgres",
            server_id="srv-1",
            server_name="nexus_pg_mcp",
            namespace="k8s",
        )
        assert tool.qualified_name == "k8s.query_postgres"

    def test_default_namespace(self):
        tool = MountedTool(
            tool_name="explore",
            server_id="s",
            server_name="n",
        )
        assert tool.qualified_name == "default.explore"


class TestToolCallRequest:
    def test_auto_id(self):
        req = ToolCallRequest(tool_name="my_tool")
        assert req.id

    def test_defaults(self):
        req = ToolCallRequest(tool_name="t")
        assert req.arguments == {}
        assert req.caller_agent == "executor"


# ─────────────────────────────────────────────
# Sandbox static analysis tests
# ─────────────────────────────────────────────

class TestSandboxStaticAnalysis:
    def setup_method(self):
        self.sandbox = Sandbox()

    def _make_spec(self, code: str) -> MCPServerSpec:
        return MCPServerSpec(
            request_id="r",
            server_name="test_mcp",
            language=ServerLanguage.PYTHON,
            source_code=code,
        )

    def test_clean_code_passes(self):
        spec = self._make_spec("def hello(): return 'world'")
        check = self.sandbox._static_analysis(spec)
        assert check.passed

    def test_os_system_blocked(self):
        spec = self._make_spec("import os\nos.system('rm -rf /')")
        check = self.sandbox._static_analysis(spec)
        assert not check.passed
        assert "os.system" in check.message

    def test_subprocess_blocked(self):
        spec = self._make_spec("import subprocess\nsubprocess.run(['ls'])")
        check = self.sandbox._static_analysis(spec)
        assert not check.passed

    def test_eval_blocked(self):
        spec = self._make_spec("result = eval(user_input)")
        check = self.sandbox._static_analysis(spec)
        assert not check.passed

    def test_exec_blocked(self):
        spec = self._make_spec("exec('import os')")
        check = self.sandbox._static_analysis(spec)
        assert not check.passed

    def test_requests_blocked(self):
        spec = self._make_spec("import requests\nrequests.get('http://evil.com')")
        check = self.sandbox._static_analysis(spec)
        assert not check.passed

    def test_all_dangerous_patterns_covered(self):
        """Verify we have at least the core dangerous patterns."""
        assert any("os.system" in p for p in DANGEROUS_PATTERNS)
        assert any("subprocess" in p for p in DANGEROUS_PATTERNS)
        assert any("eval" in p for p in DANGEROUS_PATTERNS)

    def test_mock_args_string(self):
        schema = {"properties": {"name": {"type": "string"}, "count": {"type": "integer"}}}
        args = self.sandbox._build_mock_args(schema)
        assert args["name"] == "mock_name"
        assert args["count"] == 1

    def test_mock_args_all_types(self):
        schema = {
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "f": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            }
        }
        args = self.sandbox._build_mock_args(schema)
        assert isinstance(args["s"], str)
        assert isinstance(args["i"], int)
        assert isinstance(args["f"], float)
        assert isinstance(args["b"], bool)
        assert isinstance(args["a"], list)
        assert isinstance(args["o"], dict)


# ─────────────────────────────────────────────
# Router registry tests (no subprocesses)
# ─────────────────────────────────────────────

class TestMCPRouter:
    def _make_mounted_spec(self, name: str = "test_mcp", ns: str = "default") -> MCPServerSpec:
        spec = MCPServerSpec(
            request_id="r",
            server_name=name,
            language=ServerLanguage.PYTHON,
            source_code="",
            tools=[
                ToolSpec(name="tool_one", description="First tool"),
                ToolSpec(name="tool_two", description="Second tool"),
            ],
            status=ServerStatus.MOUNTED,
            namespace=ns,
            mounted_at=time.time(),
        )
        return spec

    @pytest.mark.asyncio
    async def test_mount_registers_tools(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec()

        names = await router.mount(spec)
        assert len(names) == 2
        assert "default.tool_one" in names
        assert "default.tool_two" in names

    @pytest.mark.asyncio
    async def test_tool_count_after_mount(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec()
        await router.mount(spec)
        assert router.tool_count == 2

    @pytest.mark.asyncio
    async def test_lookup_by_qualified_name(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec(ns="prod")
        await router.mount(spec)

        tool = router.get_tool("prod.tool_one")
        assert tool is not None
        assert tool.tool_name == "tool_one"

    @pytest.mark.asyncio
    async def test_lookup_by_unqualified_name(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec()
        await router.mount(spec)

        tool = router.get_tool("tool_one")  # shorthand
        assert tool is not None

    @pytest.mark.asyncio
    async def test_mount_requires_mounted_status(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec()
        spec.status = ServerStatus.GENERATED  # not yet validated

        with pytest.raises(ValueError, match="Cannot mount"):
            await router.mount(spec)

    @pytest.mark.asyncio
    async def test_unmount_removes_tools(self):
        sandbox = MagicMock()
        sandbox.teardown = AsyncMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec()
        await router.mount(spec)
        assert router.tool_count == 2

        await router.unmount(spec.id)
        assert router.tool_count == 0

    @pytest.mark.asyncio
    async def test_call_unknown_tool_returns_error(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)

        response = await router.call(ToolCallRequest(
            tool_name="nonexistent_tool",
            arguments={},
        ))
        assert not response.success
        assert "not found" in response.error

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        assert router.list_tool_names() == []

    @pytest.mark.asyncio
    async def test_list_tools_after_mount(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        spec = self._make_mounted_spec(ns="k8s")
        await router.mount(spec)
        names = router.list_tool_names()
        assert "k8s.tool_one" in names

    @pytest.mark.asyncio
    async def test_namespace_filter(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        await router.mount(self._make_mounted_spec(name="mcp1", ns="dev"))
        await router.mount(self._make_mounted_spec(name="mcp2", ns="prod"))

        dev_tools = router.list_tool_names(namespace="dev")
        prod_tools = router.list_tool_names(namespace="prod")
        assert all("dev." in t for t in dev_tools)
        assert all("prod." in t for t in prod_tools)

    @pytest.mark.asyncio
    async def test_stats_structure(self):
        sandbox = MagicMock()
        router = MCPRouter(sandbox)
        stats = router.stats()
        assert "total_tools" in stats
        assert "total_servers" in stats
        assert "tools" in stats


# ─────────────────────────────────────────────
# Synthesizer stub path tests (no LLM)
# ─────────────────────────────────────────────

class TestSynthesizerStubs:
    def setup_method(self):
        self.synthesizer = SynthesizerAgent.__new__(SynthesizerAgent)

    def test_stub_implementations_one_tool(self):
        tools = [ToolSpec(
            name="query_db",
            description="Run a DB query",
            parameters=[ToolParameter(name="sql", type="string", description="SQL")],
        )]
        code, dispatch = self.synthesizer._stub_implementations(tools)
        assert "def impl_query_db" in code
        assert "query_db" in dispatch
        assert dispatch["query_db"] == "impl_query_db"

    def test_stub_dispatch_has_all_tools(self):
        tools = [
            ToolSpec(name="tool_a", description="A"),
            ToolSpec(name="tool_b", description="B"),
            ToolSpec(name="tool_c", description="C"),
        ]
        code, dispatch = self.synthesizer._stub_implementations(tools)
        assert set(dispatch.keys()) == {"tool_a", "tool_b", "tool_c"}

    def test_stub_no_params(self):
        tools = [ToolSpec(name="ping", description="Ping the system")]
        code, dispatch = self.synthesizer._stub_implementations(tools)
        assert "def impl_ping" in code
        assert "return 'Mock result for ping'" in code

    def test_build_tools_manifest(self):
        tools = [ToolSpec(
            name="get_status",
            description="Get system status",
            parameters=[
                ToolParameter(name="host", type="string", description="Host"),
                ToolParameter(name="port", type="integer", description="Port",
                              required=False),
            ],
        )]
        manifest = self.synthesizer._build_tools_manifest(tools)
        assert len(manifest) == 1
        tool_entry = manifest[0]
        assert tool_entry["name"] == "get_status"
        assert "host" in tool_entry["inputSchema"]["properties"]
        assert "host" in tool_entry["inputSchema"]["required"]
        assert "port" not in tool_entry["inputSchema"]["required"]

    def test_fallback_tools(self):
        req = SynthesisRequest(
            gap_description="Need something for legacy COBOL mainframe",
            target_system="IBM z/OS",
        )
        tools = self.synthesizer._fallback_tools(req)
        assert len(tools) == 1
        assert tools[0].name == "explore_system"

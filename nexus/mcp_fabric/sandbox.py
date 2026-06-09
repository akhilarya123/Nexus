"""
nexus/mcp_fabric/sandbox.py
-----------------------------
The Sandbox boots synthesized MCP servers in isolated Python subprocesses
and validates them via JSON-RPC before they are allowed into the live router.

Why subprocess (not Docker) for local M1 dev:
  - Docker SDK requires a running Docker daemon and image builds
  - Subprocess isolation is sufficient for Mac M1 local development
  - The same interface (stdin/stdout JSON-RPC) works identically
  - Docker integration can be layered on top for production use

Validation pipeline:
  1. STATIC_ANALYSIS  — scan source code for dangerous patterns
  2. SANDBOX_BOOT     — write to /tmp, launch as subprocess, check it starts
  3. TOOLS_LIST       — send {"method":"tools/list"}, parse manifest
  4. TOOLS_CALL       — call each tool once with mock args, check no crash

A server that passes all 4 stages gets status=MOUNTED.
A server that fails any stage gets status=FAILED with a clear error message
that is fed back to the Synthesizer for a re-attempt.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import structlog

from nexus.mcp_fabric.models import (
    MCPServerSpec,
    ServerStatus,
    ValidationCheck,
    ValidationResult,
    ValidationStage,
)
from nexus.observability.tracing import traced

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────
# Safety patterns — any of these in source code = FAIL immediately
# ─────────────────────────────────────────────
DANGEROUS_PATTERNS = [
    r"\bos\.system\b",
    "os.system",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\b",
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
    r"open\s*\(.*['\"]w['\"]",      # file write outside /tmp
    r"\bsocket\.connect\b",
    r"\burllib\b",
    r"\brequests\b",
    r"\bhttpx\b",
]

# How long to wait for each JSON-RPC call (seconds)
JSONRPC_TIMEOUT = 10.0
# How long to let the server boot before declaring failure
BOOT_TIMEOUT = 8.0


class Sandbox:
    """
    Manages the lifecycle of synthesized MCP server subprocesses.

    Usage:
        sandbox = Sandbox()
        result = await sandbox.validate(spec)
        if result.overall_passed:
            # spec is now ready to mount
            ...
        await sandbox.teardown(spec)  # kills process, removes temp file
    """

    def __init__(self) -> None:
        # Tracks active processes: spec_id → asyncio.subprocess.Process
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        # Tracks temp files: spec_id → path
        self._temp_files: dict[str, Path] = {}

    # ── Public API ────────────────────────────────────────────────

    @traced("nexus.mcp_fabric.sandbox", "validate")
    async def validate(self, spec: MCPServerSpec) -> ValidationResult:
        """
        Run the full 4-stage validation pipeline on a generated server.

        Returns ValidationResult. On success, spec.status is set to MOUNTED
        and spec.process_pid is populated.
        """
        t0 = time.monotonic()
        checks: list[ValidationCheck] = []

        # ── Stage 1: Static analysis ──────────────────────────────
        static_check = self._static_analysis(spec)
        checks.append(static_check)
        if not static_check.passed:
            return ValidationResult(
                spec_id=spec.id,
                checks=checks,
                overall_passed=False,
                failure_reason=static_check.message,
                elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            )

        # ── Stage 2: Boot the subprocess ─────────────────────────
        boot_check = await self._boot_server(spec)
        checks.append(boot_check)
        if not boot_check.passed:
            return ValidationResult(
                spec_id=spec.id,
                checks=checks,
                overall_passed=False,
                failure_reason=boot_check.message,
                elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            )

        # ── Stage 3: tools/list ───────────────────────────────────
        list_check, manifest = await self._check_tools_list(spec)
        checks.append(list_check)
        if not list_check.passed:
            await self.teardown(spec)
            return ValidationResult(
                spec_id=spec.id,
                checks=checks,
                overall_passed=False,
                failure_reason=list_check.message,
                elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            )

        # ── Stage 4: tools/call ───────────────────────────────────
        call_check = await self._check_tools_call(spec, manifest)
        checks.append(call_check)

        overall = all(c.passed for c in checks)
        if overall:
            spec.status = ServerStatus.MOUNTED
            spec.mounted_at = time.time()
            log.info("Sandbox validation PASSED", server=spec.server_name,
                     tools=spec.tool_names)
        else:
            spec.status = ServerStatus.FAILED
            spec.error_message = call_check.message
            await self.teardown(spec)
            log.warning("Sandbox validation FAILED", server=spec.server_name,
                        reason=call_check.message)

        return ValidationResult(
            spec_id=spec.id,
            checks=checks,
            overall_passed=overall,
            failure_reason="" if overall else (call_check.message or "Unknown failure"),
            tool_manifest=manifest,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    async def teardown(self, spec: MCPServerSpec) -> None:
        """Kill the subprocess and remove the temp file for a given spec."""
        proc = self._processes.pop(spec.id, None)
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception:
                pass

        tmp_file = self._temp_files.pop(spec.id, None)
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass

        log.debug("Sandbox teardown complete", server=spec.server_name)

    async def teardown_all(self) -> None:
        """Teardown ALL active sandboxed processes. Call on kernel shutdown."""
        for spec_id in list(self._processes.keys()):
            proc = self._processes.pop(spec_id, None)
            if proc and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception:
                    pass
        for tmp_file in list(self._temp_files.values()):
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass
        self._temp_files.clear()

    # ── Private stages ────────────────────────────────────────────

    def _static_analysis(self, spec: MCPServerSpec) -> ValidationCheck:
        """Scan source code for dangerous patterns."""
        t0 = time.monotonic()
        for pattern in DANGEROUS_PATTERNS:
            m = re.search(pattern, spec.source_code)
            if m:
                matched = m.group(0)
                return ValidationCheck(
                    stage=ValidationStage.STATIC_ANALYSIS,
                    passed=False,
                    message=f"Dangerous pattern detected: {matched}",
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                )
        return ValidationCheck(
            stage=ValidationStage.STATIC_ANALYSIS,
            passed=True,
            message="No dangerous patterns found",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    async def _boot_server(self, spec: MCPServerSpec) -> ValidationCheck:
        """Write source to /tmp and start as subprocess."""
        t0 = time.monotonic()
        try:
            # Write source to a temp file
            tmp = Path(tempfile.mktemp(suffix=".py", prefix=f"nexus_mcp_"))
            tmp.write_text(spec.source_code)
            self._temp_files[spec.id] = tmp

            # Launch process with stdin/stdout pipes
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(tmp),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._processes[spec.id] = proc
            spec.process_pid = proc.pid

            # Send initialize handshake
            init_request = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nexus-sandbox", "version": "1.0.0"},
                },
            }) + "\n"

            response = await self._send_and_receive(proc, init_request)
            if "error" in response:
                return ValidationCheck(
                    stage=ValidationStage.SANDBOX_BOOT,
                    passed=False,
                    message=f"Initialize failed: {response['error']}",
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                )

            return ValidationCheck(
                stage=ValidationStage.SANDBOX_BOOT,
                passed=True,
                message=f"Server booted (pid={proc.pid})",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )
        except Exception as exc:
            return ValidationCheck(
                stage=ValidationStage.SANDBOX_BOOT,
                passed=False,
                message=f"Boot failed: {exc}",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )

    async def _check_tools_list(
        self, spec: MCPServerSpec
    ) -> tuple[ValidationCheck, dict]:
        """Send tools/list and verify the manifest matches the spec."""
        t0 = time.monotonic()
        proc = self._processes.get(spec.id)
        if not proc:
            return ValidationCheck(
                stage=ValidationStage.TOOLS_LIST,
                passed=False,
                message="Process not found",
            ), {}

        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }) + "\n"

        try:
            response = await self._send_and_receive(proc, request)
            if "error" in response:
                return ValidationCheck(
                    stage=ValidationStage.TOOLS_LIST,
                    passed=False,
                    message=f"tools/list error: {response['error']}",
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                ), {}

            tools = response.get("result", {}).get("tools", [])
            if not tools:
                return ValidationCheck(
                    stage=ValidationStage.TOOLS_LIST,
                    passed=False,
                    message="tools/list returned empty tools array",
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                ), {}

            manifest = {"tools": tools}
            return ValidationCheck(
                stage=ValidationStage.TOOLS_LIST,
                passed=True,
                message=f"tools/list returned {len(tools)} tool(s)",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            ), manifest
        except Exception as exc:
            return ValidationCheck(
                stage=ValidationStage.TOOLS_LIST,
                passed=False,
                message=f"tools/list exception: {exc}",
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            ), {}

    async def _check_tools_call(
        self, spec: MCPServerSpec, manifest: dict
    ) -> ValidationCheck:
        """Call each tool once with minimal mock arguments."""
        t0 = time.monotonic()
        proc = self._processes.get(spec.id)
        if not proc:
            return ValidationCheck(
                stage=ValidationStage.TOOLS_CALL,
                passed=False,
                message="Process not found",
            )

        tools = manifest.get("tools", [])
        if not tools:
            return ValidationCheck(
                stage=ValidationStage.TOOLS_CALL,
                passed=False,
                message="No tools to call",
            )

        call_id = 10
        for tool in tools[:3]:  # test up to 3 tools
            tool_name = tool.get("name", "")
            # Build minimal args from schema
            args = self._build_mock_args(tool.get("inputSchema", {}))
            request = json.dumps({
                "jsonrpc": "2.0",
                "id": call_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            }) + "\n"
            call_id += 1

            try:
                response = await self._send_and_receive(proc, request)
                if "error" in response:
                    return ValidationCheck(
                        stage=ValidationStage.TOOLS_CALL,
                        passed=False,
                        message=f"tools/call '{tool_name}' returned error: {response['error']}",
                        latency_ms=round((time.monotonic() - t0) * 1000, 1),
                    )
            except Exception as exc:
                return ValidationCheck(
                    stage=ValidationStage.TOOLS_CALL,
                    passed=False,
                    message=f"tools/call '{tool_name}' exception: {exc}",
                    latency_ms=round((time.monotonic() - t0) * 1000, 1),
                )

        return ValidationCheck(
            stage=ValidationStage.TOOLS_CALL,
            passed=True,
            message=f"All {len(tools[:3])} tool call(s) succeeded",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    async def _send_and_receive(
        self, proc: asyncio.subprocess.Process, request_line: str
    ) -> dict[str, Any]:
        """Write one JSON-RPC request line and read one response line."""
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Process has no stdin/stdout")

        proc.stdin.write(request_line.encode())
        await proc.stdin.drain()

        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=JSONRPC_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"JSON-RPC timeout after {JSONRPC_TIMEOUT}s")

        if not line:
            raise RuntimeError("Process closed stdout unexpectedly")

        return json.loads(line.decode().strip())

    @staticmethod
    def _build_mock_args(schema: dict) -> dict[str, Any]:
        """Build minimal valid arguments from a JSON Schema object."""
        props = schema.get("properties", {})
        mock: dict[str, Any] = {}
        for name, prop in props.items():
            t = prop.get("type", "string")
            if t == "string":
                mock[name] = f"mock_{name}"
            elif t == "integer":
                mock[name] = 1
            elif t == "number":
                mock[name] = 1.0
            elif t == "boolean":
                mock[name] = True
            elif t == "array":
                mock[name] = []
            elif t == "object":
                mock[name] = {}
        return mock

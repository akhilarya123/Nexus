from unittest.mock import AsyncMock

from nexus.mcp_fabric.models import ToolCallResponse
from nexus.orchestration.models import ActionResult, PlanStep
from nexus.orchestration.kernel import OrchestrationKernel


def test_kernel_synthesizes_and_calls_tool_on_gap():
    kernel = OrchestrationKernel()

    kernel.planner.plan_next_step = lambda state, critic: PlanStep(
        action_type="TOOL_GAP",
        parameters={
            "needs_new_tool": True,
            "tool_gap_description": "Need to query an undocumented Cassandra cluster",
            "target_system": "Apache Cassandra 4.x",
            "tools": [
                {"name": "cql_query", "description": "Run CQL"},
            ],
            "call_tool_name": "cql_query",
            "call_tool_arguments": {"query": "SELECT * FROM system.local"},
        },
    )
    kernel.executor.execute = lambda step: ActionResult(
        success=False,
        output="Tool gap detected",
    )
    kernel.critic.evaluate_state = lambda state: 1.0
    kernel.fabric.synthesize_and_mount = AsyncMock(return_value=["default.cql_query"])
    kernel.fabric.call_tool = AsyncMock(
        return_value=ToolCallResponse(
            request_id="req-1",
            tool_name="cql_query",
            success=True,
            result="mock rows",
        )
    )

    final_state = kernel.run_exploration(
        initial_context="Starting context",
        max_steps=1,
    )

    assert final_state.current_context == "mock rows"
    assert final_state.history == [
        "TOOL_GAP",
        "Synthesized tools: ['default.cql_query']",
        "Tool call cql_query: True",
    ]
    kernel.fabric.synthesize_and_mount.assert_awaited_once()
    kernel.fabric.call_tool.assert_awaited_once()
import asyncio
import os

from .models import AgentState
from .planner import GlobalPlannerAgent
from .execution import ExecutionAgent
from .critic import CriticAgent
from nexus.mcp_fabric import MCPFabric, ToolSpec

class OrchestrationKernel:
    def __init__(self, enable_dynamic_tools: bool | None = None):
        self.planner = GlobalPlannerAgent()
        self.executor = ExecutionAgent()
        self.critic = CriticAgent()
        if enable_dynamic_tools is None:
            env_value = os.getenv("NEXUS_ENABLE_DYNAMIC_TOOLS", "1").strip().lower()
            enable_dynamic_tools = env_value not in {"0", "false", "no", "off"}
        self.enable_dynamic_tools = enable_dynamic_tools
        self.fabric = MCPFabric()

    def run_exploration(self, initial_context: str, max_steps: int = 50):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_exploration_async(initial_context, max_steps))

        raise RuntimeError(
            "run_exploration() cannot be called from a running event loop; "
            "use await run_exploration_async(...) instead."
        )

    async def run_exploration_async(self, initial_context: str, max_steps: int = 50):
        state = AgentState(current_context=initial_context)
        attempted_gap_signatures: set[str] = set()

        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")

            # 1. Plan
            action = self.planner.plan_next_step(state, self.critic)
            print(f"[*] Planner decided: {action.action_type}")

            # 2. Execute
            result = self.executor.execute(action)

            # 3. Update State
            state.history.append(action.action_type)
            state.current_context = result.output
            state.step_count += 1

            # 3b. If the planner/executor surfaced a capability gap, synthesize tools on demand.
            if self.enable_dynamic_tools and self._needs_tool_gap(action.action_type, action.parameters, result):
                gap_description = (
                    action.parameters.get("tool_gap_description")
                    or result.output
                    or f"Need a new tool for {action.action_type}"
                )
                target_system = action.parameters.get("target_system", "")
                tool_specs = self._coerce_tool_specs(action.parameters.get("tools"))
                gap_signature = self._build_gap_signature(
                    action_type=action.action_type,
                    gap_description=gap_description,
                    target_system=target_system,
                )

                # Avoid repeatedly re-synthesizing the same gap in long loops.
                if gap_signature not in attempted_gap_signatures:
                    attempted_gap_signatures.add(gap_signature)
                    await self.fabric.synthesize_and_mount(
                        gap_description=gap_description,
                        target_system=target_system,
                        tools=tool_specs,
                    )

                call_tool_name = action.parameters.get("call_tool_name")
                if call_tool_name:
                    call_tool_arguments = action.parameters.get("call_tool_arguments", {})
                    resp = await self.fabric.call_tool(call_tool_name, call_tool_arguments)
                    state.current_context = resp.result if resp.success else resp.error

            # 4. Asynchronous Evaluation (Synchronous here for simplicity)
            health_score = self.critic.evaluate_state(state)
            print(f"[*] Critic Health Score: {health_score}")

            if health_score < 0.3:
                print("[!] Critic detected loop or hallucination. Pruning path and backtracking.")
                # Basic backtracking mechanism for the simulation
                state.current_context = "Backtracked to safe state."

        return state

    @staticmethod
    def _needs_tool_gap(action_type: str, parameters, result) -> bool:
        if action_type in {"TOOL_GAP", "MCP_TOOL_GAP", "NEEDS_TOOL"}:
            return True

        if not isinstance(parameters, dict):
            return False

        if parameters.get("needs_new_tool"):
            return True

        # Treat explicit gap/tool metadata as an intent to synthesize tools.
        explicit_gap_keys = {
            "tool_gap_description",
            "tools",
            "call_tool_name",
            "target_system",
        }
        if any(k in parameters for k in explicit_gap_keys):
            return True

        return False

    @staticmethod
    def _coerce_tool_specs(raw_tools) -> list[ToolSpec]:
        if not raw_tools:
            return []

        tool_specs: list[ToolSpec] = []
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if isinstance(item, ToolSpec):
                    tool_specs.append(item)
                elif isinstance(item, dict) and item.get("name") and item.get("description") is not None:
                    tool_specs.append(ToolSpec(**item))

        return tool_specs

    @staticmethod
    def _build_gap_signature(action_type: str, gap_description: str, target_system: str) -> str:
        return "|".join([
            action_type.strip(),
            (gap_description or "").strip(),
            (target_system or "").strip(),
        ])
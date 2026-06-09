from .models import PlanStep, ActionResult

class ExecutionAgent:
    def execute(self, step: PlanStep) -> ActionResult:
        # Mocking real system interaction for the 50-step simulation
        print(f"[Execution] Running {step.action_type} with {step.parameters}")
        
        # Simulate environment response
        if step.action_type == "EXPLORE_NODE":
            return ActionResult(success=True, output=f"Found topology data at {step.parameters.get('target')}")
        elif step.action_type == "QUERY_DB":
            return ActionResult(success=True, output="Retrieved schemas.")
        else:
            return ActionResult(success=False, output="Unknown action execution.")
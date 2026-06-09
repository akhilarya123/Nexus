from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PlanStep(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ActionResult(BaseModel):
    success: bool
    output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    current_context: str
    history: List[str] = Field(default_factory=list)
    step_count: int = 0

class MCTSNode:
    def __init__(self, state: AgentState, parent: Optional['MCTSNode'] = None, action_taken: Optional[PlanStep] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.value: float = 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0 # Simplified for local execution
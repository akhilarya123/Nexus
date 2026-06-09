from .models import AgentState, MCTSNode, PlanStep
from .mcts import MCTSTree
from .llm_client import query_ollama

class GlobalPlannerAgent:
    def __init__(self):
        self.mcts = MCTSTree()

    def expand(self, node: MCTSNode):
        prompt = f"""Given the context: {node.state.current_context}. 
        Propose 2 distinct micro-actions to explore this system. 
        Format as exactly: ACTION_TYPE|param_key:param_val"""
        
        response = query_ollama(prompt)
        # Naive parsing for simulation
        lines = response.split('\n')
        for line in lines:
            if '|' in line:
                action, params_raw = line.split('|', 1)
                params = {params_raw.split(':')[0]: params_raw.split(':')[1]} if ':' in params_raw else {}
                
                new_state = AgentState(
                    current_context=f"Executed {action}",
                    history=node.state.history + [action],
                    step_count=node.state.step_count + 1
                )
                child = MCTSNode(state=new_state, parent=node, action_taken=PlanStep(action_type=action.strip(), parameters=params))
                node.children.append(child)

    def plan_next_step(self, current_state: AgentState, critic, iterations=3) -> PlanStep:
        root = MCTSNode(state=current_state)
        
        for _ in range(iterations):
            leaf = self.mcts.select(root)
            self.expand(leaf)
            if leaf.children:
                simulated_node = leaf.children[0]
                reward = critic.evaluate_state(simulated_node.state)
                self.mcts.backpropagate(simulated_node, reward)

        best_node = self.mcts.best_child(root)
        return best_node.action_taken if best_node else PlanStep(action_type="IDLE", parameters={})
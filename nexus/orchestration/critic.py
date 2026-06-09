from .models import AgentState
from .llm_client import query_ollama

class CriticAgent:
    def evaluate_state(self, state: AgentState) -> float:
        """Returns a reward score between 0.0 and 1.0"""
        prompt = f"""Evaluate this system state for logical errors or hallucination loops. 
        History: {state.history}
        Current: {state.current_context}
        Score the state strictly from 0.0 (failure/loop) to 1.0 (productive progression). Reply ONLY with a float number."""
        
        response = query_ollama(prompt, temperature=0.1)
        try:
            score = float(response)
            return min(max(score, 0.0), 1.0) # Clamp between 0 and 1
        except ValueError:
            return 0.1 # Penalty for unparseable output
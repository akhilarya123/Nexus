import pytest
from nexus.orchestration.models import AgentState, MCTSNode
from nexus.orchestration.mcts import MCTSTree

def test_mcts_selection_and_backprop():
    tree = MCTSTree()
    root = MCTSNode(state=AgentState(current_context="root"))
    
    # Create artificial children
    child1 = MCTSNode(state=AgentState(current_context="child1"), parent=root)
    child2 = MCTSNode(state=AgentState(current_context="child2"), parent=root)
    root.children.extend([child1, child2])
    
    # Test Backprop
    tree.backpropagate(child1, 1.0)
    assert child1.visits == 1
    assert child1.value == 1.0
    assert root.visits == 1
    
    # Test UCT Selection (should pick child2 since child1 was visited)
    selected = tree.select(root)
    assert selected == child2
import math
import random
from typing import Optional
from .models import MCTSNode

class MCTSTree:
    def __init__(self, exploration_weight: float = 1.41):
        self.exploration_weight = exploration_weight

    def select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded and node.children:
            node = self.best_child(node)
        return node

    def best_child(self, node: MCTSNode) -> MCTSNode | None:
        if not node.children:
            return None

        best_score = float('-inf')
        best_c = None
        for child in node.children:
            if child.visits == 0:
                return child
            # UCB1 formula
            exploit = child.value / child.visits
            explore = self.exploration_weight * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_c = child
        return best_c or random.choice(node.children)

    def backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
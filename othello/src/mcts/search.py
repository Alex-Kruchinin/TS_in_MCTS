from __future__ import annotations

import math
import random

from game import Action, OthelloState
from mcts.node import MCTSNode


class UCTSearch:
    """Run classical UCT-MCTS for Othello"""

    def __init__(
        self,
        simulations: int = 500,
        exploration_constant: float = math.sqrt(2.0),
        *,
        seed: int | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("Simulations must be greater than zero.")
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")

        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self._rng = random.Random(seed)

    def build_tree(self, root_state: OthelloState) -> MCTSNode:
        """Run all simulations and return the populated root node."""

        if root_state.is_terminal():
            raise ValueError("Cannot search from a terminal state.")

        root = MCTSNode(root_state)

        for _ in range(self.simulations):
            node = self._select(root)

            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand(self._rng)

            terminal_state = self._rollout(node.state)
            node.backpropagate(terminal_state)

        return root

    def choose_action(self, root_state: OthelloState) -> tuple[Action, MCTSNode]:
        """Return the selected action and completed tree"""

        root = self.build_tree(root_state)
        action = self.best_root_action(root)
        return action, root

    def best_root_action(self, root: MCTSNode) -> Action:
        """Choose by visits, then mean reward"""

        if not root.children:
            raise ValueError("The root has no expanded children.")

        best_key = max(
            (child.visits, child.mean_value)
            for child in root.children
        )
        best_children = [
            child
            for child in root.children
            if (child.visits, child.mean_value) == best_key
        ]
        selected = self._rng.choice(best_children)

        if selected.action is None:
            raise RuntimeError("A root child must have an incoming action.")
        return selected.action

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root

        while (
            not node.is_terminal
            and node.is_fully_expanded
            and node.children
        ):
            node = node.best_child(
                self.exploration_constant,
                self._rng,
            )

        return node

    def _rollout(self, start_state: OthelloState) -> OthelloState:
        state = start_state
        action_count = 0
        maximum_actions = state.rows * state.cols * 2

        while not state.is_terminal():
            if action_count >= maximum_actions:
                raise RuntimeError(
                    "Rollout exceeded its safety action limit"
                )

            legal_actions = state.legal_actions()
            action = self._rng.choice(legal_actions)
            state = state.apply(action)
            action_count += 1

        return state

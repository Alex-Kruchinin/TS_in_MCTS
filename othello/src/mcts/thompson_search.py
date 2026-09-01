from __future__ import annotations

import math
import random

from game import Action, OthelloState
from mcts.thompson_node import ThompsonMCTSNode


class ThompsonSearch:
    """Classical Othello MCTS with Thompson tree selection"""

    def __init__(
        self,
        simulations: int = 500,
        *,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("Simulations must be greater than zero.")
        if not math.isfinite(alpha_prior) or alpha_prior <= 0:
            raise ValueError("Alpha prior must be finite and greater than zero.")
        if not math.isfinite(beta_prior) or beta_prior <= 0:
            raise ValueError("Beta prior must be finite and greater than zero.")

        self.simulations = simulations
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._rng = random.Random(seed)

    def build_tree(self, root_state: OthelloState) -> ThompsonMCTSNode:
        if root_state.is_terminal():
            raise ValueError("Cannot search from a terminal state.")

        root = ThompsonMCTSNode(
            root_state,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
        )

        for _ in range(self.simulations):
            node = self._select(root)

            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand(self._rng)

            terminal_state = self._rollout(node.state)
            node.backpropagate(terminal_state)

        return root

    def choose_action(
        self,
        root_state: OthelloState,
    ) -> tuple[Action, ThompsonMCTSNode]:
        root = self.build_tree(root_state)
        return self.best_root_action(root), root

    def best_root_action(self, root: ThompsonMCTSNode) -> Action:
        """Choose by visits, then mean reward, as in classical UCT."""

        if not root.children:
            raise ValueError("The root has no expanded children.")

        best_key = max(
            (child.visits, child.mean_value)
            for child in root.children
        )
        candidates = [
            child
            for child in root.children
            if (child.visits, child.mean_value) == best_key
        ]
        selected = self._rng.choice(candidates)

        if selected.action is None:
            raise RuntimeError("A root child must have an incoming action.")
        return selected.action

    def _select(self, root: ThompsonMCTSNode) -> ThompsonMCTSNode:
        node = root

        while (
            not node.is_terminal
            and node.is_fully_expanded
            and node.children
        ):
            node = node.sample_child(self._rng)

        return node

    def _rollout(self, start_state: OthelloState) -> OthelloState:
        state = start_state
        action_count = 0
        maximum_actions = state.rows * state.cols * 2

        while not state.is_terminal():
            if action_count >= maximum_actions:
                raise RuntimeError(
                    "Rollout exceeded its safety action limit."
                )

            action = self._rng.choice(state.legal_actions())
            state = state.apply(action)
            action_count += 1

        return state

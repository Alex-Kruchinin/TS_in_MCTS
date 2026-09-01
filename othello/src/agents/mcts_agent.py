from __future__ import annotations

import math

from game import Action, OthelloState
from mcts import MCTSNode, UCTSearch


class MCTSAgent:
    """Othello agent that selects actions using UCT-MCTS."""

    def __init__(
        self,
        simulations: int = 500,
        exploration_constant: float = math.sqrt(2.0),
        *,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self._search = UCTSearch(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed,
        )
        self._name = name or f"MCTSAgent({simulations})"
        self.last_root: MCTSNode | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def simulations(self) -> int:
        return self._search.simulations

    @property
    def exploration_constant(self) -> float:
        return self._search.exploration_constant

    def choose_action(self, state: OthelloState) -> Action:
        if state.is_terminal():
            raise ValueError("Cannot choose an action from a terminal state.")

        action, root = self._search.choose_action(state)
        self.last_root = root
        return action

from __future__ import annotations

from game import Action, OthelloState
from mcts import ThompsonMCTSNode, ThompsonSearch


class ThompsonMCTSAgent:
    """Classical Othello MCTS using Thompson Sampling during selection."""

    def __init__(
        self,
        simulations: int = 500,
        *,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self._search = ThompsonSearch(
            simulations=simulations,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            seed=seed,
        )
        self._name = name or f"ThompsonMCTSAgent({simulations})"
        self.last_root: ThompsonMCTSNode | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def simulations(self) -> int:
        return self._search.simulations

    @property
    def alpha_prior(self) -> float:
        return self._search.alpha_prior

    @property
    def beta_prior(self) -> float:
        return self._search.beta_prior

    def choose_action(self, state: OthelloState) -> Action:
        if state.is_terminal():
            raise ValueError("Cannot choose an action from a terminal state.")

        action, root = self._search.choose_action(state)
        self.last_root = root
        return action

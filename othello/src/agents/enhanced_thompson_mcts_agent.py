from __future__ import annotations

from game import Action, OthelloState
from mcts import EnhancedThompsonNode, EnhancedThompsonSearch


class EnhancedThompsonMCTSAgent:
    """Enhanced Othello MCTS with Thompson Sampling tree selection."""

    def __init__(
        self,
        simulations: int = 500,
        *,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        rollout_epsilon: float = 0.15,
        expansion_epsilon: float = 0.05,
        rollout_depth_limit: int = 16,
        guided_expansion: bool = True,
        heuristic_rollouts: bool = True,
        root_safety: bool = True,
        root_lookahead: bool = True,
        heuristic_cutoff: bool = True,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        self._search = EnhancedThompsonSearch(
            simulations=simulations,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            rollout_epsilon=rollout_epsilon,
            expansion_epsilon=expansion_epsilon,
            rollout_depth_limit=rollout_depth_limit,
            guided_expansion=guided_expansion,
            heuristic_rollouts=heuristic_rollouts,
            root_safety=root_safety,
            root_lookahead=root_lookahead,
            heuristic_cutoff=heuristic_cutoff,
            seed=seed,
        )
        self._name = name or f"EnhancedThompsonMCTSAgent({simulations})"
        self.last_root: EnhancedThompsonNode | None = None

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

    @property
    def rollout_epsilon(self) -> float:
        return self._search.rollout_epsilon

    @property
    def expansion_epsilon(self) -> float:
        return self._search.expansion_epsilon

    @property
    def rollout_depth_limit(self) -> int:
        return self._search.rollout_depth_limit

    @property
    def guided_expansion(self) -> bool:
        return self._search.guided_expansion

    @property
    def heuristic_rollouts(self) -> bool:
        return self._search.heuristic_rollouts

    @property
    def root_safety(self) -> bool:
        return self._search.root_safety

    @property
    def root_lookahead(self) -> bool:
        return self._search.root_lookahead

    @property
    def heuristic_cutoff(self) -> bool:
        return self._search.heuristic_cutoff

    def choose_action(self, state: OthelloState) -> Action:
        if state.is_terminal():
            raise ValueError("Cannot choose an action from a terminal state.")

        action, root = self._search.choose_action(state)
        self.last_root = root
        return action

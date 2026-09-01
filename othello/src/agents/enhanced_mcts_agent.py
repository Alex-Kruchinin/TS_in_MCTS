from __future__ import annotations

import math

from game import Action, OthelloState
from mcts import EnhancedMCTSNode, EnhancedUCTSearch


class EnhancedMCTSAgent:
    """Othello UCT agent with switchable features for optional ablation studies."""

    def __init__(
        self,
        simulations: int = 500,
        exploration_constant: float = math.sqrt(2.0),
        *,
        progressive_bias_weight: float = 0.75,
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
        self._search = EnhancedUCTSearch(
            simulations=simulations,
            exploration_constant=exploration_constant,
            progressive_bias_weight=progressive_bias_weight,
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
        self._name = name or f"EnhancedMCTSAgent({simulations})"
        self.last_root: EnhancedMCTSNode | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def simulations(self) -> int:
        return self._search.simulations

    @property
    def exploration_constant(self) -> float:
        return self._search.exploration_constant

    @property
    def progressive_bias_weight(self) -> float:
        return self._search.progressive_bias_weight

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

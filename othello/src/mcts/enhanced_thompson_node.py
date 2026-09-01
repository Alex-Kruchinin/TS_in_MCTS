from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from game import Action, Disc, OthelloState
from mcts.heuristics import OthelloHeuristicPolicy


@dataclass(slots=True)
class EnhancedThompsonNode:
    """Thompson node used with Othello-guided expansion and rollouts."""

    state: OthelloState
    parent: EnhancedThompsonNode | None = None
    action: Action | None = None
    prior_score: float = 0.0
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    children: list[EnhancedThompsonNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list[Action] = field(init=False)

    def __post_init__(self) -> None:
        if self.parent is None and self.action is not None:
            raise ValueError("A root node cannot have an incoming action.")
        if self.parent is not None and self.action is None:
            raise ValueError("A non-root node must record its incoming action.")
        if not math.isfinite(self.alpha_prior) or self.alpha_prior <= 0:
            raise ValueError("Alpha prior must be finite and greater than zero.")
        if not math.isfinite(self.beta_prior) or self.beta_prior <= 0:
            raise ValueError("Beta prior must be finite and greater than zero.")

        self.untried_actions = list(self.state.legal_actions())

    @property
    def player_just_moved(self) -> Disc:
        return self.state.current_player.opponent

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def alpha(self) -> float:
        return self.alpha_prior + self.total_value

    @property
    def beta(self) -> float:
        return self.beta_prior + self.visits - self.total_value

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    @property
    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def expand(
        self,
        policy: OthelloHeuristicPolicy,
        rng: random.Random,
        *,
        expansion_epsilon: float,
    ) -> EnhancedThompsonNode:
        if self.is_terminal:
            raise ValueError("A terminal node cannot be expanded.")
        if not self.untried_actions:
            raise ValueError("This node is already fully expanded.")

        action = policy.choose_action(
            self.state,
            self.untried_actions,
            rng,
            epsilon=expansion_epsilon,
        )
        self.untried_actions.remove(action)
        child = EnhancedThompsonNode(
            state=self.state.apply(action),
            parent=self,
            action=action,
            prior_score=policy.normalised_prior(self.state, action),
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
        )
        self.children.append(child)
        return child

    def sample_child(self, rng: random.Random) -> EnhancedThompsonNode:
        if not self.children:
            raise ValueError("Cannot select a child from a leaf node.")

        sampled = [
            (rng.betavariate(child.alpha, child.beta), child)
            for child in self.children
        ]
        best_sample = max(sample for sample, _ in sampled)
        candidates = [
            child
            for sample, child in sampled
            if math.isclose(
                sample,
                best_sample,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]
        return rng.choice(candidates)

    def backpropagate_values(
        self,
        *,
        black_value: float,
        white_value: float,
    ) -> None:
        """Update posteriors with terminal or heuristic fractional rewards."""

        if not 0.0 <= black_value <= 1.0:
            raise ValueError("Black value must be between zero and one.")
        if not 0.0 <= white_value <= 1.0:
            raise ValueError("White value must be between zero and one.")

        node: EnhancedThompsonNode | None = self
        while node is not None:
            reward = (
                black_value
                if node.player_just_moved is Disc.BLACK
                else white_value
            )
            node.visits += 1
            node.total_value += reward
            node = node.parent

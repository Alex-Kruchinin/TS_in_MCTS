from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from game import Action, Disc, OthelloState
from mcts.heuristics import OthelloHeuristicPolicy


@dataclass(slots=True)
class EnhancedMCTSNode:
    """MCTS node with an Othello action prior for progressive bias."""

    state: OthelloState
    parent: EnhancedMCTSNode | None = None
    action: Action | None = None
    prior_score: float = 0.0
    children: list[EnhancedMCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list[Action] = field(init=False)

    def __post_init__(self) -> None:
        if self.parent is None and self.action is not None:
            raise ValueError("A root node cannot have an incoming action.")
        if self.parent is not None and self.action is None:
            raise ValueError("A non-root node must record its incoming action.")
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
    ) -> EnhancedMCTSNode:
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
        child = EnhancedMCTSNode(
            state=self.state.apply(action),
            parent=self,
            action=action,
            prior_score=policy.normalised_prior(self.state, action),
        )
        self.children.append(child)
        return child

    def best_child(
        self,
        exploration_constant: float,
        progressive_bias_weight: float,
        rng: random.Random,
    ) -> EnhancedMCTSNode:
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")
        if progressive_bias_weight < 0:
            raise ValueError("Progressive-bias weight cannot be negative.")
        if not self.children:
            raise ValueError("Cannot select a child from a leaf node.")

        parent_visits = max(1, self.visits)
        scored: list[tuple[float, EnhancedMCTSNode]] = []

        for child in self.children:
            if child.visits == 0:
                score = math.inf
            else:
                exploration = exploration_constant * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                progressive_bias = (
                    progressive_bias_weight
                    * child.prior_score
                    / (child.visits + 1)
                )
                score = child.mean_value + exploration + progressive_bias
            scored.append((score, child))

        best_score = max(score for score, _ in scored)
        candidates = [child for score, child in scored if score == best_score]
        return rng.choice(candidates)

    def backpropagate(self, terminal_state: OthelloState) -> None:
        if not terminal_state.is_terminal():
            raise ValueError("Backpropagation requires a terminal state.")
        self.backpropagate_values(
            black_value=terminal_state.result_for(Disc.BLACK),
            white_value=terminal_state.result_for(Disc.WHITE),
        )

    def backpropagate_values(
        self,
        *,
        black_value: float,
        white_value: float,
    ) -> None:
        if not 0.0 <= black_value <= 1.0:
            raise ValueError("Black value must be between zero and one.")
        if not 0.0 <= white_value <= 1.0:
            raise ValueError("White value must be between zero and one.")

        node: EnhancedMCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.total_value += (
                black_value
                if node.player_just_moved is Disc.BLACK
                else white_value
            )
            node = node.parent

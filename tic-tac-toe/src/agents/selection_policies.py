from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.mcts_node import MCTSNode


class TreeSelectionPolicy(Protocol):
    """Interface for selecting one existing child during MCTS selection."""

    def select_child(
        self,
        node: MCTSNode,
        rng: Random,
    ) -> MCTSNode:
        """Select and return one child of a fully expanded node."""
        ...


def _random_best(
    scored_children: list[tuple[float, MCTSNode]],
    rng: Random,
) -> MCTSNode:
    """Return a random child among those tied for the best score."""

    best_score = max(score for score, _ in scored_children)
    best_children = [
        child
        for score, child in scored_children
        if math.isclose(score, best_score)
    ]

    return rng.choice(best_children)


@dataclass(frozen=True, slots=True)
class UCTSelectionPolicy:
    """Select children with UCT.

    UCT balances high rewards with less-visited moves:

    ``mean reward + C * sqrt(ln(parent visits) / child visits)``
    """

    exploration_constant: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.exploration_constant):
            raise ValueError("exploration_constant must be finite.")

        if self.exploration_constant < 0:
            raise ValueError(
                "exploration_constant cannot be negative."
            )

    def select_child(
        self,
        node: MCTSNode,
        rng: Random,
    ) -> MCTSNode:
        """Return the child with the highest UCT score."""

        if not node.children:
            raise ValueError("Cannot select a child from an empty node.")

        def uct_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return math.inf

            if node.visits <= 0:
                return child.mean_value

            exploration = self.exploration_constant * math.sqrt(
                math.log(node.visits) / child.visits
            )

            return child.mean_value + exploration

        scored_children = [
            (uct_score(child), child)
            for child in node.children.values()
        ]

        return _random_best(scored_children, rng)


@dataclass(frozen=True, slots=True)
class ProgressiveBiasUCTSelectionPolicy:
    """UCT with a heuristic bonus that decreases with visits.

    ``UCT score + heuristic_weight * H(child) / (child.visits + 1)``
    """

    exploration_constant: float = math.sqrt(2.0)
    heuristic_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.exploration_constant):
            raise ValueError("exploration_constant must be finite.")

        if self.exploration_constant < 0:
            raise ValueError(
                "exploration_constant cannot be negative."
            )

        if not math.isfinite(self.heuristic_weight):
            raise ValueError("heuristic_weight must be finite.")

        if self.heuristic_weight < 0:
            raise ValueError("heuristic_weight cannot be negative.")

    def select_child(
        self,
        node: MCTSNode,
        rng: Random,
    ) -> MCTSNode:
        """Return the child with the highest progressive-bias UCT score."""

        if not node.children:
            raise ValueError("Cannot select a child from an empty node.")

        def score(child: MCTSNode) -> float:
            if child.visits == 0:
                return math.inf

            if node.visits <= 0:
                uct_part = child.mean_value
            else:
                exploration = self.exploration_constant * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                uct_part = child.mean_value + exploration

            progressive_bias = (
                self.heuristic_weight
                * child.heuristic_value
                / (child.visits + 1.0)
            )

            return uct_part + progressive_bias

        scored_children = [
            (score(child), child)
            for child in node.children.values()
        ]

        return _random_best(scored_children, rng)


@dataclass(frozen=True, slots=True)
class ThompsonSamplingSelectionPolicy:
    """Select the highest sample drawn from each child's Beta posterior."""

    def select_child(
        self,
        node: MCTSNode,
        rng: Random,
    ) -> MCTSNode:
        """Return the child with the highest Thompson sample."""

        if not node.children:
            raise ValueError("Cannot select a child from an empty node.")

        sampled_children = [
            (rng.betavariate(child.alpha, child.beta), child)
            for child in node.children.values()
        ]

        return _random_best(sampled_children, rng)

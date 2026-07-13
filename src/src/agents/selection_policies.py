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


@dataclass(frozen=True, slots=True)
class UCTSelectionPolicy:
    """
    Select children using Upper Confidence Bounds applied to Trees.

    UCT balances:
        exploitation -> prefer moves with a high observed mean reward;
        exploration  -> revisit moves with fewer samples.

    Formula:
        mean reward + C * sqrt(ln(parent visits) / child visits)
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

        # Normally every expanded child has at least one visit because a
        # rollout follows immediately after expansion. Treating an unvisited
        # child as infinity also keeps this method safe when used directly.
        def uct_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return math.inf

            if node.visits <= 0:
                return child.mean_value  # child’s updated average reward across every rollout that has passed through
                                         # total reward / visits
            exploration = self.exploration_constant * math.sqrt(
                math.log(node.visits) / child.visits
            )

            return child.mean_value + exploration

        scored_children = [
            (uct_score(child), child)
            for child in node.children.values()
        ]

        best_score = max(score for score, _ in scored_children)
        best_children = [
            child
            for score, child in scored_children
            if math.isclose(score, best_score)
        ]

        # Random tie-breaking avoids always favouring whichever move happened
        # to appear first in a dictionary while remaining reproducible by seed.
        return rng.choice(best_children)

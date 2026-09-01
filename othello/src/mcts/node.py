from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from game import Action, Disc, OthelloState


@dataclass(slots=True)
class MCTSNode:
    """One state in an Othello MCTS tree

    Values use the perspective of the player who made the incoming action.
    """

    state: OthelloState
    parent: MCTSNode | None = None
    action: Action | None = None
    children: list[MCTSNode] = field(default_factory=list)
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
        """Player who produced this node's state."""

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

    def expand(self, rng: random.Random) -> MCTSNode:
        """Create one child by selecting one previously untried action."""

        if self.is_terminal:
            raise ValueError("A terminal node cannot be expanded.")
        if not self.untried_actions:
            raise ValueError("This node is already fully expanded.")

        action_index = rng.randrange(len(self.untried_actions))
        action = self.untried_actions.pop(action_index)
        child = MCTSNode(
            state=self.state.apply(action),
            parent=self,
            action=action,
        )
        self.children.append(child)
        return child

    def best_child(
        self,
        exploration_constant: float,
        rng: random.Random,
    ) -> MCTSNode:
        """Select the child with the highest UCT score"""

        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")
        if not self.children:
            raise ValueError("Cannot select a child from a leaf node.")

        parent_visits = max(1, self.visits)
        scored_children: list[tuple[float, MCTSNode]] = []

        for child in self.children:
            if child.visits == 0:
                score = math.inf
            else:
                exploration = exploration_constant * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                score = child.mean_value + exploration

            scored_children.append((score, child))

        best_score = max(score for score, _ in scored_children)
        best_children = [
            child
            for score, child in scored_children
            if score == best_score
        ]
        return rng.choice(best_children)

    def backpropagate(self, terminal_state: OthelloState) -> None:
        """Update this node and every ancestor with a terminal result"""

        if not terminal_state.is_terminal():
            raise ValueError("Backpropagation requires a terminal state.")

        node: MCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.total_value += terminal_state.result_for(
                node.player_just_moved
            )
            node = node.parent

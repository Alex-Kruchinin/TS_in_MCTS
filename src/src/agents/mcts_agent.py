from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from src.agents.mcts_node import MCTSNode
from src.agents.selection_policies import (
    TreeSelectionPolicy,
    UCTSelectionPolicy,
)
from src.games.tic_tac_toe import Move, TicTacToeState


@dataclass(frozen=True, slots=True)
class MCTSAgent:
    """
    Monte Carlo Tree Search agent.

    The agent performs a fixed number of MCTS iterations whenever it must
    choose a real move. Each iteration contains:
        1. selection;
        2. expansion;
        3. simulation (rollout);
        4. backpropagation.

    UCT is the default tree-selection policy. Later, a Thompson Sampling
    policy can be supplied without replacing the rest of this class.
    """

    simulations: int = 1000  # number of iterations
    selection_policy: TreeSelectionPolicy = field(
        default_factory=UCTSelectionPolicy
    )

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be a positive integer.")

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """
        Search from the supplied state and return the final selected move.

        The final policy uses the robust child rule: choose the root child
        with the highest visit count. Mean value is used as a tie-breaker.
        """

        if state.is_terminal():
            raise ValueError(
                "MCTSAgent cannot choose a move from a terminal state."
            )

        root = self.search(state, rng)  # build the search tree

        if not root.children:
            raise RuntimeError("MCTS search produced no root children.")

        # find the best child
        best_key = max(
            (child.visits, child.mean_value)
            for child in root.children.values()
        )

        best_children = [
            child
            for child in root.children.values()
            if (child.visits, child.mean_value) == best_key
        ]

        selected_child = rng.choice(best_children) # break ties randomly

        if selected_child.move is None:
            raise RuntimeError("Selected MCTS child has no incoming move.")

        return selected_child.move

    def search(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> MCTSNode:
        """Build and return an MCTS tree rooted at the supplied state."""

        if state.is_terminal():
            raise ValueError("Cannot search from a terminal state.")

        root = MCTSNode(state=state)

        for _ in range(self.simulations):
            node = root

            # 1. SELECTION
            # Follow existing children while the node is non-terminal and
            # has no untried legal moves remaining.
            while (
                not node.state.is_terminal()
                and node.is_fully_expanded()
            ):
                node = self.selection_policy.select_child(node, rng)  # select child based on policy

            # 2. EXPANSION
            # Add one previously untried move to the tree.
            if not node.state.is_terminal():
                node = node.expand(rng)

            # 3. SIMULATION / PLAYOUT
            terminal_state = self._rollout(node.state, rng)

            # 4. BACKPROPAGATION
            self._backpropagate(node, terminal_state)

        return root

    @staticmethod
    def _rollout(
        state: TicTacToeState,
        rng: Random,
    ) -> TicTacToeState:
        """
        Play uniformly random legal moves until a terminal state is reached.

        Rollout states are temporary and never change the real match state.
        """

        rollout_state = state

        while not rollout_state.is_terminal():
            move = rng.choice(rollout_state.legal_moves())
            rollout_state = rollout_state.apply_move(move)

        return rollout_state

    @staticmethod
    def _backpropagate(
        node: MCTSNode,
        terminal_state: TicTacToeState,
    ) -> None:
        """Update every node from the rollout start back to the root."""

        current: MCTSNode | None = node

        while current is not None:
            current.update(terminal_state)
            current = current.parent

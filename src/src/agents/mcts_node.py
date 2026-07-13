from __future__ import annotations

from random import Random
from typing import Optional

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


class MCTSNode:
    """
    The representation of a node in a Monte Carlo Tree Search tree.

    Except for the root, it also stores the move that was applied
    to the parent state to reach this state.

    Statistics are stored from the perspective of the player who made
    the move leading into this node. This means that, when a player is
    choosing among the node's children, every child's mean value is from
    that player's perspective.
    """

    def __init__(
        self,
        state: TicTacToeState,  # refers to the state of the board
        parent: Optional[MCTSNode] = None,
        move: Optional[Move] = None,  # i.e Move(1, 1)
    ) -> None:
        self.state = state
        self.parent = parent
        self.move = move

        # Existing branches that have already been added to the tree.
        self.children: dict[Move, MCTSNode] = {}

        # Legal moves that have not yet been expanded into child nodes.
        self.untried_moves: list[Move] = list(state.legal_moves())

        # Number of completed MCTS iterations that passed through this node.
        self.visits = 0

        # Sum of simulation rewards from the perspective of the player
        # who made the move leading into this node.
        self.value_sum = 0.0

    @property
    def player_just_moved(self) -> Mark:
        """
        Return the player who made the move leading into this state.

        TicTacToeState stores the player who moves NEXT. Therefore, the
        previous player is obtained by changing the sign:
            X (1) -> O (-1)
            O (-1) -> X (1)

        For the root there is no actual incoming move, but its value is
        not used to choose the final action. Its visit count is still
        needed by the UCT exploration term.
        """

        return Mark(-int(self.state.player_to_move))

    @property
    def mean_value(self) -> float:
        """Return the average reward observed at this node."""

        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    def is_fully_expanded(self) -> bool:
        """Return True when every legal move has a child node."""

        return not self.untried_moves

    def expand(self, rng: Random) -> MCTSNode:
        """
        Expand one randomly selected untried move and return its child.

        Only one new child is created in a single MCTS iteration.
        """

        if self.state.is_terminal():
            raise ValueError("A terminal node cannot be expanded.")

        if not self.untried_moves:
            raise ValueError("This node is already fully expanded.")

        move_index = rng.randrange(len(self.untried_moves))
        move = self.untried_moves.pop(move_index)
        child_state = self.state.apply_move(move)

        child = MCTSNode(
            state=child_state,
            parent=self,
            move=move,
        )

        self.children[move] = child
        return child

    def update(self, terminal_state: TicTacToeState) -> None:
        """
        Update this node using the result of one completed simulation.

        Reward convention from player_just_moved's perspective:
            win  -> 1.0
            draw -> 0.5
            loss -> 0.0
        """

        if not terminal_state.is_terminal():
            raise ValueError(
                "MCTSNode can only be updated from a terminal state."
            )

        reward = terminal_state.reward_for(self.player_just_moved)

        self.visits += 1
        self.value_sum += reward

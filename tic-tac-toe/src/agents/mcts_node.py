from __future__ import annotations

from random import Random
from typing import Optional

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


class MCTSNode:
    """One state in an MCTS tree.

    Each non-root node stores the move from its parent. Values use the
    perspective of the player who made that move.

    Thompson nodes also store Beta parameters: alpha for success evidence and
    beta for failure evidence.
    """

    def __init__(
        self,
        state: TicTacToeState,
        parent: Optional[MCTSNode] = None,
        move: Optional[Move] = None,
        heuristic_value: float = 0.0,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move = move

        # Expanded child branches.
        self.children: dict[Move, MCTSNode] = {}

        # Legal moves not yet expanded.
        self.untried_moves: list[Move] = list(state.legal_moves())

        # Searches that visited this node.
        self.visits = 0

        # Total reward for the player who made the incoming move.
        self.value_sum = 0.0

        # Thompson posterior, initialised from the chosen Beta prior.
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.alpha = initial_alpha
        self.beta = initial_beta

        # Move heuristic used by progressive bias.
        self.heuristic_value = heuristic_value

    @property
    def player_just_moved(self) -> Mark:
        """Return the player who made the incoming move.

        The state stores the next player, so changing its sign gives the
        previous player. The root value is not used to choose a move.
        """

        return Mark(-int(self.state.player_to_move))

    @property
    def mean_value(self) -> float:
        """Return the average reward observed at this node."""

        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    @property
    def posterior_mean(self) -> float:
        """Return the mean of the Thompson Sampling Beta posterior."""

        return self.alpha / (self.alpha + self.beta)

    def is_fully_expanded(self) -> bool:
        """Return True when every legal move has a child node."""

        return not self.untried_moves

    def expand(
        self,
        rng: Random,
        move: Optional[Move] = None,
        heuristic_value: float = 0.0,
    ) -> MCTSNode:
        """Expand an untried move and return its child.

        Select randomly when ``move`` is omitted. A supplied move must still
        be untried.
        """

        if self.state.is_terminal():
            raise ValueError("A terminal node cannot be expanded.")

        if not self.untried_moves:
            raise ValueError("This node is already fully expanded.")

        if move is None:
            move_index = rng.randrange(len(self.untried_moves))
            move = self.untried_moves.pop(move_index)
        else:
            if move not in self.untried_moves:
                raise ValueError("The supplied move is not untried.")
            self.untried_moves.remove(move)

        child_state = self.state.apply_move(move)

        child = MCTSNode(
            state=child_state,
            parent=self,
            move=move,
            heuristic_value=heuristic_value,
            initial_alpha=self.initial_alpha,
            initial_beta=self.initial_beta,
        )

        self.children[move] = child
        return child

    def update(self, terminal_state: TicTacToeState) -> None:
        """Update this node from a completed rollout.

        Reward convention from player_just_moved's perspective:
            win  -> 1.0
            draw -> 0.5
            loss -> 0.0

        The reward updates both the MCTS value and Thompson posterior.
        """

        if not terminal_state.is_terminal():
            raise ValueError(
                "MCTSNode can only be updated from a terminal state."
            )

        reward = terminal_state.reward_for(self.player_just_moved)

        self.visits += 1
        self.value_sum += reward

        # Treat a draw as half success and half failure.
        self.alpha += reward
        self.beta += 1.0 - reward

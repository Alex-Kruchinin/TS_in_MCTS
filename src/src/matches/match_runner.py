from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional

from src.agents.base_agent import Agent
from src.games.tic_tac_toe import Mark, Move, TicTacToeState


@dataclass(frozen=True, slots=True)
class MatchResult:
    """
    Information produced after one complete Tic-Tac-Toe match.

    Attributes:
        final_state:
            Terminal game state reached at the end of the match.

        winner:
            Mark.X or Mark.O when a player wins, otherwise None for a
            draw.

        move_history:
            Every move in the order in which it was played.

        seed:
            Random seed used for the match. Saving it makes an experiment
            reproducible.
    """

    final_state: TicTacToeState
    winner: Optional[Mark]
    move_history: tuple[Move, ...]
    seed: Optional[int]

    @property
    def number_of_moves(self) -> int:
        """Return the total number of moves played in the match."""

        return len(self.move_history)

    @property
    def is_draw(self) -> bool:
        """Return True when the completed match ended in a draw."""

        return self.final_state.is_draw()


@dataclass(frozen=True, slots=True)
class MatchRunner:
    """
    Run complete scalable Tic-Tac-Toe matches between two agents.

    MatchRunner is responsible for the match procedure, not move quality:
        1. create the initial board;
        2. identify whose turn it is;
        3. ask that player's agent to choose a move;
        4. verify and apply the move;
        5. repeat until the game is terminal;
        6. return the match result.
    """

    # If no configuration is provided, use standard Tic-Tac-Toe.
    # So runner = MatchRunner() creates a 3x3 Tic-Tac-Toe game.
    # While runner = MatchRunner(rows=5, cols=5, win_length=4,) - sets a 5x5 game. With 4 win length

    rows: int = 3
    cols: int = 3
    win_length: int = 3

    def play(
        self,
        agent_x: Agent,
        agent_o: Agent,
        seed: Optional[int] = None,
    ) -> MatchResult:
        """
        Play one match and return its complete result.

        X always makes the first move because TicTacToeState.new()
        initially sets player_to_move to Mark.X.
        """

        rng = Random(seed)

        state = TicTacToeState.new(
            rows=self.rows,
            cols=self.cols,
            win_length=self.win_length,
        )

        move_history: list[Move] = []

        # Main loop: while game has not finished, do...

        while not state.is_terminal():
            # Select the agent belonging to the player whose turn it is.
            if state.player_to_move == Mark.X:
                current_agent = agent_x
            else:
                current_agent = agent_o

            # The agent receives the complete immutable state and returns
            # the move it wants to play.
            move = current_agent.choose_move(state, rng)

            # Additional check: MatchRunner verifies the return even though a correct agent
            # should only return legal moves. This catches agent bugs early.
            if not state.is_legal_move(move):
                raise ValueError(
                    f"Agent for {state.player_to_move.name} returned "
                    f"illegal move ({move.row}, {move.col})."
                )

            move_history.append(move) # a move is added to the history, and a new immutable state is created
            state = state.apply_move(move)

        return MatchResult(
            final_state=state,
            winner=state.winner(),
            move_history=tuple(move_history),
            seed=seed,
        )

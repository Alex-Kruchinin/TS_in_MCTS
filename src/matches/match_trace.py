from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


@dataclass(frozen=True, slots=True)
class TraceStep:
    """
    One recorded move inside a complete match trace.

    A TraceStep stores both the board before and after the move. This makes
    the trace useful for debugging and for explaining a game step by step.
    """

    move_number: int
    player: Mark
    agent_name: str
    move: Move
    state_before_move: TicTacToeState
    state_after_move: TicTacToeState

    def to_text(self) -> str:
        """Return a readable text representation of this single step."""

        lines = [
            f"Move {self.move_number}",
            f"Player: {self.player.name}",
            f"Agent:  {self.agent_name}",
            f"Move:   row {self.move.row}, col {self.move.col}",
            "",
            str(self.state_after_move),
        ]

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MatchTrace:
    """
    Human-readable record of one complete match.

    MatchRunner is still responsible for running the game. MatchTrace only
    stores and formats what happened during that game.
    """

    initial_state: TicTacToeState
    steps: tuple[TraceStep, ...]
    final_state: TicTacToeState
    winner: Optional[Mark]
    seed: Optional[int]
    agent_x_name: str
    agent_o_name: str

    @property
    def number_of_moves(self) -> int:
        """Return the number of moves recorded in the trace."""

        return len(self.steps)

    @property
    def is_draw(self) -> bool:
        """Return True when the traced game ended in a draw."""

        return self.final_state.is_draw()

    def result_text(self) -> str:
        """Return a compact text description of the final result."""

        if self.winner == Mark.X:
            return "Winner: X"

        if self.winner == Mark.O:
            return "Winner: O"

        return "Result: draw"

    def to_text(self) -> str:
        """
        Return the full match trace as readable multi-line text.

        This is intended for console output, dissertation appendices, or
        saving to a plain text file.
        """

        separator = "=" * 72

        lines = [
            separator,
            "Single Tic-Tac-Toe match trace",
            f"Board: {self.initial_state.rows}x{self.initial_state.cols}",
            f"Win length: {self.initial_state.win_length}",
            f"X agent: {self.agent_x_name}",
            f"O agent: {self.agent_o_name}",
            f"Seed: {self.seed}",
            separator,
            "",
            "Initial board:",
            "",
            str(self.initial_state),
        ]

        for step in self.steps:
            lines.extend([
                "",
                "-" * 72,
                step.to_text(),
            ])

        lines.extend([
            "",
            separator,
            "Final result:",
            self.result_text(),
            f"Moves played: {self.number_of_moves}",
            separator,
        ])

        return "\n".join(lines)

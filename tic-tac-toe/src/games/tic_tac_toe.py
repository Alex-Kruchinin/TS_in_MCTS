from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Mark(IntEnum):
    """Values stored in a board cell: empty, X or O."""

    EMPTY = 0
    X = 1
    O = -1


# Immutable row and column coordinates.
@dataclass(frozen=True, slots=True)
class Move:
    """row and column coordinates for one square."""

    row: int
    col: int


@dataclass(frozen=True, slots=True)
class TicTacToeState:
    """Immutable state of a scalable Tic-Tac-Toe game.

    The board is a flat tuple. Applying a move returns a new state so MCTS
    branches can keep separate positions.
    """

    rows: int
    cols: int
    win_length: int
    board: tuple[Mark, ...]
    player_to_move: Mark = Mark.X
    last_move: Optional[Move] = None
    moves_played: int = 0

    def __post_init__(self) -> None:
        """Board dimensions checks"""

        if self.rows <= 0:
            raise ValueError("The number of rows must be positive.")

        if self.cols <= 0:
            raise ValueError("The number of columns must be positive.")

        # A winning line must fit along at least one board dimension.
        if not 1 <= self.win_length <= max(self.rows, self.cols):
            raise ValueError(
                "win_length must be between 1 and the largest "
                "board dimension."
            )

        expected_board_size = self.rows * self.cols

        if len(self.board) != expected_board_size:
            raise ValueError(
                f"Expected {expected_board_size} board cells, "
                f"but received {len(self.board)}."
            )

        if self.player_to_move not in (Mark.X, Mark.O):
            raise ValueError(
                "player_to_move must be either Mark.X or Mark.O."
            )

        if not 0 <= self.moves_played <= expected_board_size:
            raise ValueError("Invalid number of moves played.")

        occupied_cells = sum(
            mark != Mark.EMPTY for mark in self.board
        )

        if occupied_cells != self.moves_played:
            raise ValueError(
                "moves_played must equal the number of occupied cells."
            )

        if self.last_move is not None and not self.is_inside_board(
            self.last_move
        ):
            raise ValueError("last_move is outside the board.")

    @classmethod
    def new(
        cls,
        rows: int = 3,
        cols: int = 3,
        win_length: int = 3,
    ) -> TicTacToeState:
        """Create an empty game with X to move."""

        return cls(
            rows=rows,
            cols=cols,
            win_length=win_length,
            # Create one empty cell per board position.
            board=(Mark.EMPTY,) * (rows * cols),
            player_to_move=Mark.X,
            last_move=None,
            moves_played=0,
        )

    def is_inside_board(self, move: Move) -> bool:
        """Return whether ``move`` is inside the board."""

        return (
            0 <= move.row < self.rows
            and 0 <= move.col < self.cols
        )

    def move_to_index(self, move: Move) -> int:
        """Convert row and column coordinates to a flat board index."""

        if not self.is_inside_board(move):
            raise ValueError(
                f"Move ({move.row}, {move.col}) is outside "
                f"the {self.rows}x{self.cols} board."
            )

        return move.row * self.cols + move.col

    def index_to_move(self, index: int) -> Move:
        """Convert a flat board index to row and column coordinates."""

        if not 0 <= index < len(self.board):
            raise ValueError(
                f"Board index {index} is outside the board."
            )

        # Split the index into row and column.
        row, col = divmod(index, self.cols)

        return Move(row=row, col=col)

    def cell_at(self, move: Move) -> Mark:
        """Return the mark at ``move``."""

        # Convert the coordinates to the flat board index.
        index = self.move_to_index(move)
        return self.board[index]

    def winner(self) -> Optional[Mark]:
        """Return the winner, or None if there is no winner.

        Only vertical, horizontal and diagonal lines through the latest move
        need to be checked.
        """

        if self.last_move is None:
            # An empty board has no winner.
            return None

        last_mark = self.cell_at(self.last_move)

        if last_mark == Mark.EMPTY:
            # Keep manually created invalid states safe.
            return None

        directions = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )

        for row_step, col_step in directions:
            # Include the latest move in calculation
            marks_in_line = 1

            # Count one side of the line
            marks_in_line += self._count_direction(
                start=self.last_move,
                row_step=row_step,
                col_step=col_step,
                mark=last_mark,
            )

            # Count the opposite side
            marks_in_line += self._count_direction(
                start=self.last_move,
                row_step=-row_step,
                col_step=-col_step,
                mark=last_mark,
            )

            if marks_in_line >= self.win_length:
                return last_mark

        return None

    def _count_direction(
        self,
        start: Move,
        row_step: int,
        col_step: int,
        mark: Mark,
    ) -> int:
        """Count consecutive matching marks in one direction."""

        count = 0
        row = start.row + row_step
        col = start.col + col_step

        while 0 <= row < self.rows and 0 <= col < self.cols:
            if self.cell_at(Move(row, col)) != mark:
                break

            count += 1
            row += row_step
            col += col_step

        return count

    def is_draw(self) -> bool:
        """Return whether the full board has no winner."""

        board_is_full = self.moves_played == self.rows * self.cols
        return board_is_full and self.winner() is None

    def is_terminal(self) -> bool:
        """Return whether the game has ended."""

        return self.winner() is not None or self.is_draw()

    def reward_for(self, player: Mark) -> float:
        """Return ``player``'s reward from a finished game.

        Reward convention:
            win  -> 1.0
            draw -> 0.5
            loss -> 0.0

        """

        if player not in (Mark.X, Mark.O):
            raise ValueError("Reward can only be requested for X or O.")

        if not self.is_terminal():
            raise ValueError(
                "A reward cannot be calculated before the game ends."
            )

        winning_player = self.winner()

        if winning_player is None:
            return 0.5

        if winning_player == player:
            return 1.0

        return 0.0

    def is_legal_move(self, move: Move) -> bool:
        """Return whether ``move`` is in an empty cell of an active game."""

        if self.is_terminal():
            return False

        if not self.is_inside_board(move):
            return False

        return self.cell_at(move) == Mark.EMPTY

    def legal_moves(self) -> tuple[Move, ...]:
        """Return all legal moves, or an empty tuple after the game ends."""

        if self.is_terminal():
            return ()

        moves = []

        # Convert every empty board index to a move
        for index, mark in enumerate(self.board):
            if mark == Mark.EMPTY:
                moves.append(self.index_to_move(index))

        return tuple(moves)

    def apply_move(self, move: Move) -> TicTacToeState:
        """Apply ``move`` and return a new state."""

        if self.is_terminal():
            raise ValueError(
                "A move cannot be applied after the game has ended."
            )

        if not self.is_inside_board(move):
            raise ValueError(
                f"Move ({move.row}, {move.col}) is outside the board."
            )

        if self.cell_at(move) != Mark.EMPTY:
            raise ValueError(
                f"Square ({move.row}, {move.col}) is occupied."
            )

        # Find the flat board index
        index = self.move_to_index(move)

        # Copy the board so the original state stays unchanged
        new_board = list(self.board)

        # Add the current player's mark
        new_board[index] = self.player_to_move

        # Negating X or O switches player
        next_player = Mark(-int(self.player_to_move))

        # Return the updated state without changing the original
        return TicTacToeState(
            rows=self.rows,
            cols=self.cols,
            win_length=self.win_length,
            board=tuple(new_board),
            player_to_move=next_player,
            last_move=move,
            moves_played=self.moves_played + 1,
        )

    def __str__(self) -> str:
        """Return the board as text."""

        symbols = {
            Mark.EMPTY: ".",
            Mark.X: "X",
            Mark.O: "O",
        }

        lines = []

        for row in range(self.rows):
            row_symbols = []

            for col in range(self.cols):
                mark = self.cell_at(Move(row, col))
                row_symbols.append(symbols[mark])

            lines.append(" ".join(row_symbols))

        return "\n".join(lines)

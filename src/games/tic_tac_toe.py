from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Mark(IntEnum):
    """
    Defines three possible square values: Empty, X, O
    """

    EMPTY = 0
    X = 1
    O = -1


"""
The dataclass creates this basic constructor automatically. 
frozen=True - an object cannot be changed after it has been created
"""
@dataclass(frozen=True, slots=True)
class Move:
    """
    A move identifies one square using a row and column.

    Rows and columns are zero-indexed.

    Examples:
        Move(0, 0) -> top-left square
        Move(1, 1) -> centre of a 3x3 board
    """

    row: int
    col: int


@dataclass(frozen=True, slots=True)
class TicTacToeState:
    """
    An immutable state of a scalable Tic-Tac-Toe game - stores all the information needed to continue playing.

    Attributes:
        rows:
            Number of rows on the board.

        cols:
            Number of columns on the board.

        win_length:
            Number of consecutive marks required to win.

        board:
            Flattened immutable representation of the board.
            For example: (0, 0, 0, 0, 0, 0, 0, 0, 0) for 3x3 board

        player_to_move:
            Player who must make the next move.

        last_move:
            Most recently applied move, or None for an empty board.

        moves_played:
            Total number of moves made so far.
    """

    rows: int
    cols: int
    win_length: int
    board: tuple[Mark, ...]
    player_to_move: Mark = Mark.X
    last_move: Optional[Move] = None
    moves_played: int = 0

    def __post_init__(self) -> None:
        """
        Validate manually constructed states.
        """

        if self.rows <= 0:
            raise ValueError("The number of rows must be positive.")

        if self.cols <= 0:
            raise ValueError("The number of columns must be positive.")

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

        if self.player_to_move == Mark.EMPTY:
            raise ValueError(
                "player_to_move must be either Mark.X or Mark.O."
            )

        if not 0 <= self.moves_played <= expected_board_size:
            raise ValueError("Invalid number of moves played.")

    @classmethod
    def new(
        cls,
        rows: int = 3,
        cols: int = 3,
        win_length: int = 3,
    ) -> TicTacToeState:
        """
        Create a new empty Tic-Tac-Toe state.

        Example:
            state = TicTacToeState.new(
                rows=5,
                cols=5,
                win_length=4,
            )
        """

        return cls(
            rows=rows,
            cols=cols,
            win_length=win_length,
            board=(Mark.EMPTY,) * (rows * cols), # creates row * cols empty cells for new game
            player_to_move=Mark.X,
            last_move=None,
            moves_played=0,
        )

    def is_inside_board(self, move: Move) -> bool:
        """
        Return True when a move is inside the board boundaries.
        """

        return (
            0 <= move.row < self.rows
            and 0 <= move.col < self.cols
        )

    def move_to_index(self, move: Move) -> int:
        """
        Convert a row-column move into a flattened board index.

        Formula:
            index = row * number_of_columns + column
        """

        if not self.is_inside_board(move):
            raise ValueError(
                f"Move ({move.row}, {move.col}) is outside "
                f"the {self.rows}x{self.cols} board."
            )

        return move.row * self.cols + move.col

    def index_to_move(self, index: int) -> Move:
        """
        Convert a flattened board index into a Move: [5] -> row 1, col 2
        """

        if not 0 <= index < len(self.board):
            raise ValueError(
                f"Board index {index} is outside the board."
            )

        """
        divmod() performs division and returns:

        the whole-number result;
        the remainder.
        """
        row, col = divmod(index, self.cols)

        return Move(row=row, col=col)

    def cell_at(self, move: Move) -> Mark:
        """
        Return the value stored (Empty, X or O) at the given board position.
        """

        index = self.move_to_index(move) # coverts (row, col) to [index]
        return self.board[index]

    def is_legal_move(self, move: Move) -> bool:
        """
        A move is legal when it is inside the board and empty.
        """

        if not self.is_inside_board(move):
            return False

        return self.cell_at(move) == Mark.EMPTY # True if a cell is empty

    def legal_moves(self) -> tuple[Move, ...]:
        """
        Return all currently legal moves.
        """

        moves = []

        """
        enumerate() gives both:

        the index;
        the value stored at that index.
        """

        for index, mark in enumerate(self.board):
            if mark == Mark.EMPTY:
                moves.append(self.index_to_move(index))

        return tuple(moves)

    def apply_move(self, move: Move) -> TicTacToeState:
        """
        Apply a move and return a NEW game state.
        The current state is not modified.
        """

        if not self.is_inside_board(move):
            raise ValueError(
                f"Move ({move.row}, {move.col}) is outside "
                f"the board."
            )

        if not self.is_legal_move(move):
            raise ValueError(
                f"Square ({move.row}, {move.col}) is occupied."
            )

        index = self.move_to_index(move) # Find the internal index

        new_board = list(self.board) # Copy the board
        new_board[index] = self.player_to_move # Place the current player’s mark

        next_player = Mark(-int(self.player_to_move)) # Change player (who does next move)

        # Create a new state
        return TicTacToeState(
            rows=self.rows,
            cols=self.cols,
            win_length=self.win_length,
            board=tuple(new_board),
            player_to_move=next_player,
            last_move=move,
            moves_played=self.moves_played + 1,
        )

        """
        The new state becomes:
        board:          X placed in the centre
        player_to_move: O
        last_move:      Move(1, 1)
        moves_played:   1
        """

    def __str__(self) -> str:
        """
        Return a readable board representation.
        """

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
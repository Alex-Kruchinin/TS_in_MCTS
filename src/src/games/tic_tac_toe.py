from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Mark(IntEnum):
    """
    Defines the three possible values stored in a board cell.

    EMPTY = an unused cell
    X = player X
    O = player O
    """

    EMPTY = 0
    X = 1
    O = -1


# @dataclass creates the constructor and comparison methods automatically.
# frozen=True means a Move cannot be changed after it has been created.
# slots=True reduces the memory used by each object.
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
    An immutable state of a scalable Tic-Tac-Toe game.

    The object stores all information needed to continue the game.
    Applying a move creates a NEW TicTacToeState instead of changing
    the existing state. This is important for MCTS because separate
    branches of the search tree must keep separate board states.

    Attributes:
        rows:
            Number of rows on the board.

        cols:
            Number of columns on the board.

        win_length:
            Number of consecutive marks required to win.

        board:
            Flattened immutable representation of the board.
            Example for an empty 3x3 board:
            (0, 0, 0, 0, 0, 0, 0, 0, 0)

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
        Validate a newly created state.

        These checks protect the game from impossible dimensions and
        inconsistent board data. They do not replace the test suite;
        tests separately verify that the methods behave correctly.
        """

        if self.rows <= 0:
            raise ValueError("The number of rows must be positive.")

        if self.cols <= 0:
            raise ValueError("The number of columns must be positive.")

        # A winning line may be horizontal or vertical, so win_length
        # cannot be larger than both dimensions.
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
            # Creates rows * cols empty cells for the new game.
            board=(Mark.EMPTY,) * (rows * cols),
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

        Example on a 3-column board:
            Move(1, 2) -> 1 * 3 + 2 -> index 5
        """

        if not self.is_inside_board(move):
            raise ValueError(
                f"Move ({move.row}, {move.col}) is outside "
                f"the {self.rows}x{self.cols} board."
            )

        return move.row * self.cols + move.col

    def index_to_move(self, index: int) -> Move:
        """
        Convert a flattened board index into a Move.

        Example on a 3-column board:
            index 5 -> Move(row=1, col=2)
        """

        if not 0 <= index < len(self.board):
            raise ValueError(
                f"Board index {index} is outside the board."
            )

        # divmod() returns the whole-number result and remainder.
        # For divmod(5, 3), row = 1 and col = 2.
        row, col = divmod(index, self.cols)

        return Move(row=row, col=col)

    def cell_at(self, move: Move) -> Mark:
        """
        Return the value stored at the given board position.
        """

        # Convert (row, column) into the internal one-dimensional index.
        index = self.move_to_index(move)
        return self.board[index]

    def winner(self) -> Optional[Mark]:
        """
        Return the winning player, or None if there is no winner.

        Only lines passing through the most recent move need to be
        examined. A new winning line can only be created by that move.

        The four possible line directions are:
            vertical             (1, 0)
            horizontal           (0, 1)
            diagonal down-right  (1, 1)
            diagonal down-left   (1, -1)
        """

        if self.last_move is None:
            # No move has been played, so nobody can have won.
            return None

        last_mark = self.cell_at(self.last_move)

        if last_mark == Mark.EMPTY:
            # This should not happen for states created by apply_move(),
            # but returning None keeps the method safe.
            return None

        directions = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )

        for row_step, col_step in directions:
            # Start with 1 because the last move itself is part of the line.
            marks_in_line = 1

            # Count matching marks on one side of the last move.
            marks_in_line += self._count_direction(
                start=self.last_move,
                row_step=row_step,
                col_step=col_step,
                mark=last_mark,
            )

            # Count matching marks on the opposite side.
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
        """
        Count consecutive matching marks in one direction.

        This is a private helper method. The leading underscore means it
        is intended for use inside TicTacToeState rather than by MCTS.
        """

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
        """
        Return True when the board is full and neither player has won.
        """

        board_is_full = self.moves_played == self.rows * self.cols
        return board_is_full and self.winner() is None

    def is_terminal(self) -> bool:
        """
        Return True when the game has ended with a win or a draw.

        MCTS will use this during selection and simulation to know when it
        must stop moving through or playing out a state.
        """

        return self.winner() is not None or self.is_draw()

    def reward_for(self, player: Mark) -> float:
        """
        Return the terminal reward from one player's perspective.

        Reward convention:
            win  -> 1.0
            draw -> 0.5
            loss -> 0.0

        MCTS calls this after a simulation reaches a terminal state.
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
        """
        A move is legal when:
            1. the game has not ended;
            2. the move is inside the board;
            3. the selected cell is empty.
        """

        if self.is_terminal():
            return False

        if not self.is_inside_board(move):
            return False

        return self.cell_at(move) == Mark.EMPTY

    def legal_moves(self) -> tuple[Move, ...]:
        """
        Return all currently legal moves.

        A terminal state returns an empty tuple, even if a winning board
        still contains empty cells. MCTS must never continue after a win.
        """

        if self.is_terminal():
            return ()

        moves = []

        # enumerate() gives both the index and value stored at that index.
        for index, mark in enumerate(self.board):
            if mark == Mark.EMPTY:
                moves.append(self.index_to_move(index))

        return tuple(moves)

    def apply_move(self, move: Move) -> TicTacToeState:
        """
        Apply a move and return a NEW game state.

        The current state is not modified.
        """

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

        # Find the internal one-dimensional index.
        index = self.move_to_index(move)

        # Copy the immutable tuple into a temporary mutable list.
        new_board = list(self.board)

        # Place the current player's mark in the copied board.
        new_board[index] = self.player_to_move

        # X is 1 and O is -1, so negating the value changes player.
        next_player = Mark(-int(self.player_to_move))

        # Return a new state. The original state still represents the
        # position before this move, which allows MCTS branches to coexist.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from game.types import Action, Disc, Move, PASS, PassMove


DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True, slots=True)
class OthelloState:
    rows: int
    cols: int
    board: tuple[Disc, ...]
    current_player: Disc = Disc.BLACK

    def __post_init__(self) -> None:
        self._validate_dimensions(self.rows, self.cols)

        if len(self.board) != self.rows * self.cols:
            raise ValueError(
                "Board length must equal rows * cols: "
                f"expected {self.rows * self.cols}, got {len(self.board)}."
            )

        if self.current_player not in (Disc.BLACK, Disc.WHITE):
            raise ValueError("Current player must be BLACK or WHITE.")

        if any(not isinstance(cell, Disc) for cell in self.board):
            raise TypeError("Every board cell must be a Disc value.")

    @classmethod
    def new(
        cls,
        rows: int = 8,
        cols: int = 8,
        *,
        first_player: Disc = Disc.BLACK,
    ) -> "OthelloState":
        cls._validate_dimensions(rows, cols)

        if first_player not in (Disc.BLACK, Disc.WHITE):
            raise ValueError("First player must be BLACK or WHITE.")

        board = [Disc.EMPTY] * (rows * cols)

        upper_row = rows // 2 - 1
        lower_row = rows // 2
        left_col = cols // 2 - 1
        right_col = cols // 2

        board[upper_row * cols + left_col] = Disc.WHITE
        board[upper_row * cols + right_col] = Disc.BLACK
        board[lower_row * cols + left_col] = Disc.BLACK
        board[lower_row * cols + right_col] = Disc.WHITE

        return cls(
            rows=rows,
            cols=cols,
            board=tuple(board),
            current_player=first_player,
        )

    @classmethod
    def from_strings(
        cls,
        lines: Sequence[str],
        *,
        current_player: Disc = Disc.BLACK,
    ) -> "OthelloState":
        if not lines:
            raise ValueError("At least one board row is required.")

        stripped_lines = tuple(line.strip() for line in lines)
        cols = len(stripped_lines[0])

        if cols == 0:
            raise ValueError("Board rows cannot be empty.")

        if any(len(line) != cols for line in stripped_lines):
            raise ValueError("All board rows must have the same length.")

        symbols = {
            ".": Disc.EMPTY,
            "B": Disc.BLACK,
            "W": Disc.WHITE,
        }

        cells: list[Disc] = []
        for row, line in enumerate(stripped_lines):
            for col, symbol in enumerate(line.upper()):
                try:
                    cells.append(symbols[symbol])
                except KeyError as error:
                    raise ValueError(
                        f"Unsupported board symbol {symbol!r} "
                        f"at row {row}, column {col}."
                    ) from error

        return cls(
            rows=len(stripped_lines),
            cols=cols,
            board=tuple(cells),
            current_player=current_player,
        )

    @staticmethod
    def _validate_dimensions(rows: int, cols: int) -> None:
        if rows < 4 or cols < 4:
            raise ValueError("Othello boards must be at least 4 x 4.")

        if rows % 2 != 0 or cols % 2 != 0:
            raise ValueError(
                "Othello board dimensions must both be even."
            )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def move_to_index(self, move: Move) -> int:
        if not self.in_bounds(move.row, move.col):
            raise ValueError(
                f"Move ({move.row}, {move.col}) is outside "
                f"a {self.rows} x {self.cols} board."
            )
        return move.row * self.cols + move.col

    def index_to_move(self, index: int) -> Move:
        if index < 0 or index >= len(self.board):
            raise ValueError(f"Board index {index} is outside the board.")
        return Move(row=index // self.cols, col=index % self.cols)

    def disc_at(self, row: int, col: int) -> Disc:
        if not self.in_bounds(row, col):
            raise ValueError(
                f"Position ({row}, {col}) is outside the board."
            )
        return self.board[row * self.cols + col]

    def captured_indices(
        self,
        move: Move,
        player: Disc | None = None,
    ) -> tuple[int, ...]:
        acting_player = self.current_player if player is None else player

        if acting_player not in (Disc.BLACK, Disc.WHITE):
            raise ValueError("Acting player must be BLACK or WHITE.")

        if not self.in_bounds(move.row, move.col):
            return ()

        move_index = move.row * self.cols + move.col
        if self.board[move_index] is not Disc.EMPTY:
            return ()

        captured: list[int] = []
        opponent = acting_player.opponent

        for row_delta, col_delta in DIRECTIONS:
            row = move.row + row_delta
            col = move.col + col_delta
            direction_capture: list[int] = []

            while (
                self.in_bounds(row, col)
                and self.disc_at(row, col) is opponent
            ):
                direction_capture.append(row * self.cols + col)
                row += row_delta
                col += col_delta

            if (
                direction_capture
                and self.in_bounds(row, col)
                and self.disc_at(row, col) is acting_player
            ):
                captured.extend(direction_capture)

        return tuple(captured)

    def is_legal_placement(
        self,
        move: Move,
        player: Disc | None = None,
    ) -> bool:
        return bool(self.captured_indices(move, player))

    def legal_placements(
        self,
        player: Disc | None = None,
    ) -> tuple[Move, ...]:
        acting_player = self.current_player if player is None else player

        if acting_player not in (Disc.BLACK, Disc.WHITE):
            raise ValueError("Acting player must be BLACK or WHITE.")

        moves: list[Move] = []
        for index, cell in enumerate(self.board):
            if cell is not Disc.EMPTY:
                continue

            move = self.index_to_move(index)
            if self.captured_indices(move, acting_player):
                moves.append(move)

        return tuple(moves)

    def legal_actions(self) -> tuple[Action, ...]:
        placements = self.legal_placements(self.current_player)
        if placements:
            return placements

        if self.legal_placements(self.current_player.opponent):
            return (PASS,)

        return ()

    def apply(self, action: Action) -> "OthelloState":
        if isinstance(action, PassMove):
            return self._apply_pass()

        if not isinstance(action, Move):
            raise TypeError("Action must be a Move or PassMove.")

        captured = self.captured_indices(
            action,
            self.current_player,
        )
        if not captured:
            raise ValueError(
                f"Move ({action.row}, {action.col}) is illegal "
                f"for {self.current_player.name}."
            )

        move_index = self.move_to_index(action)
        new_board = list(self.board)
        new_board[move_index] = self.current_player

        for index in captured:
            new_board[index] = self.current_player

        return OthelloState(
            rows=self.rows,
            cols=self.cols,
            board=tuple(new_board),
            current_player=self.current_player.opponent,
        )

    def _apply_pass(self) -> "OthelloState":
        if self.legal_placements(self.current_player):
            raise ValueError(
                "Passing is illegal while the current player "
                "has a legal placement."
            )

        if not self.legal_placements(self.current_player.opponent):
            raise ValueError(
                "Passing is illegal because the game is already terminal."
            )

        return OthelloState(
            rows=self.rows,
            cols=self.cols,
            board=self.board,
            current_player=self.current_player.opponent,
        )

    def is_terminal(self) -> bool:
        return (
            not self.legal_placements(Disc.BLACK)
            and not self.legal_placements(Disc.WHITE)
        )

    def disc_count(self, disc: Disc) -> int:
        if disc not in (Disc.BLACK, Disc.WHITE, Disc.EMPTY):
            raise ValueError("Unknown disc value.")
        return self.board.count(disc)

    def counts(self) -> dict[Disc, int]:
        return {
            Disc.BLACK: self.disc_count(Disc.BLACK),
            Disc.WHITE: self.disc_count(Disc.WHITE),
            Disc.EMPTY: self.disc_count(Disc.EMPTY),
        }

    def winner(self) -> Disc | None:
        if not self.is_terminal():
            raise ValueError(
                "The winner is unavailable before the game is terminal."
            )

        black_count = self.disc_count(Disc.BLACK)
        white_count = self.disc_count(Disc.WHITE)

        if black_count > white_count:
            return Disc.BLACK
        if white_count > black_count:
            return Disc.WHITE
        return None

    def result_for(self, player: Disc) -> float:
        if player not in (Disc.BLACK, Disc.WHITE):
            raise ValueError("Result perspective must be BLACK or WHITE.")

        winning_player = self.winner()
        if winning_player is None:
            return 0.5
        if winning_player is player:
            return 1.0
        return 0.0

    def iter_rows(self) -> Iterable[tuple[Disc, ...]]:
        for row in range(self.rows):
            start = row * self.cols
            yield self.board[start : start + self.cols]

    def __str__(self) -> str:
        board_rows = (
            " ".join(cell.symbol for cell in row)
            for row in self.iter_rows()
        )
        return "\n".join(board_rows)

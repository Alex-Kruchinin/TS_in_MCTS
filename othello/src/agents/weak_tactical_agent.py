from __future__ import annotations

import random

from game import Action, Disc, Move, OthelloState, PASS


class WeakTacticalAgent:
    """Lightweight Othello opponent for parameter calibration for classical UCT and Thompson Sampling

    It prioritises corners, avoids dangerous corner-adjacent squares, prefers
    edges and then immediate flips. Also has Random moves control.
    """

    def __init__(
        self,
        *,
        random_move_probability: float = 0.25,
        seed: int | None = None,
        name: str = "WeakTacticalAgent",
    ) -> None:
        if not 0.0 <= random_move_probability <= 1.0:
            raise ValueError(
                "random_move_probability must be between 0 and 1."
            )

        self.random_move_probability = random_move_probability
        self._rng = random.Random(seed)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choose_action(self, state: OthelloState) -> Action:
        if state.is_terminal():
            raise ValueError(
                "WeakTacticalAgent cannot choose from a terminal state."
            )

        legal_actions = state.legal_actions()

        if legal_actions == (PASS,):
            return PASS

        moves = list(legal_actions)

        # 1. Always take a corner
        corners = set(self._corners(state))
        corner_moves = [move for move in moves if move in corners]
        if corner_moves:
            return self._rng.choice(corner_moves)

        # 2. Sometimes choose a random move
        if self._rng.random() < self.random_move_probability:
            return self._rng.choice(moves)

        dangerous_x, dangerous_c = self._danger_squares(state)

        # 3. Avoid unsafe X-squares
        non_x_moves = [
            move for move in moves
            if move not in dangerous_x
        ]
        if non_x_moves:
            moves = non_x_moves

        # 4. Avoid unsafe C-squares
        non_c_moves = [
            move for move in moves
            if move not in dangerous_c
        ]
        if non_c_moves:
            moves = non_c_moves

        # 5. Prefer an edge
        edge_moves = [
            move for move in moves
            if self._is_edge(state, move)
        ]
        if edge_moves:
            moves = edge_moves

        # 6. Maximise immediate flips
        flip_counts = {
            move: len(
                state.captured_indices(
                    move,
                    state.current_player,
                )
            )
            for move in moves
        }

        best_flips = max(flip_counts.values())
        best_moves = [
            move
            for move, flips in flip_counts.items()
            if flips == best_flips
        ]

        # 7. Break ties with the seeded generator
        return self._rng.choice(best_moves)

    @staticmethod
    def _corners(state: OthelloState) -> tuple[Move, Move, Move, Move]:
        return (
            Move(0, 0),
            Move(0, state.cols - 1),
            Move(state.rows - 1, 0),
            Move(state.rows - 1, state.cols - 1),
        )

    @staticmethod
    def _is_edge(state: OthelloState, move: Move) -> bool:
        return (
            move.row in (0, state.rows - 1)
            or move.col in (0, state.cols - 1)
        )

    @staticmethod
    def _danger_squares(
        state: OthelloState,
    ) -> tuple[set[Move], set[Move]]:
        """Return X- and C-squares belonging to currently empty corners."""

        last_row = state.rows - 1
        last_col = state.cols - 1

        corner_layout = (
            (
                Move(0, 0),
                Move(1, 1),
                (Move(0, 1), Move(1, 0)),
            ),
            (
                Move(0, last_col),
                Move(1, last_col - 1),
                (Move(0, last_col - 1), Move(1, last_col)),
            ),
            (
                Move(last_row, 0),
                Move(last_row - 1, 1),
                (Move(last_row - 1, 0), Move(last_row, 1)),
            ),
            (
                Move(last_row, last_col),
                Move(last_row - 1, last_col - 1),
                (
                    Move(last_row - 1, last_col),
                    Move(last_row, last_col - 1),
                ),
            ),
        )

        dangerous_x: set[Move] = set()
        dangerous_c: set[Move] = set()

        for corner, x_square, c_squares in corner_layout:
            if state.disc_at(corner.row, corner.col) is Disc.EMPTY:
                dangerous_x.add(x_square)
                dangerous_c.update(c_squares)

        return dangerous_x, dangerous_c

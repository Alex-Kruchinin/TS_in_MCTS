from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


@dataclass(frozen=True, slots=True)
class WeakTacticalAgent:
    """Weaker rule-based opponent for parameter tuning.

    It takes wins, blocks losses and otherwise prefers central squares.
    It omits the stronger line heuristic used by TacticalAgent
    """

    centre_weight: float = 1.0

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Return a legal move selected by weak tactical rules."""

        legal_moves = state.legal_moves()

        if not legal_moves:
            raise ValueError(
                "WeakTacticalAgent cannot choose a move from a terminal state."
            )

        # 1. Take an immediate win
        winning_moves = self.immediate_winning_moves(
            state=state,
            player=state.player_to_move,
        )
        if winning_moves:
            return rng.choice(winning_moves)

        # 2. Block an immediate loss
        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self.immediate_winning_moves(
            state=state,
            player=opponent,
        )
        if blocking_moves:
            return rng.choice(blocking_moves)

        # 3. Prefer a central square
        best_score = max(self.centre_score(state, move) for move in legal_moves)
        best_moves = [
            move
            for move in legal_moves
            if self.centre_score(state, move) == best_score
        ]
        return rng.choice(best_moves)

    @staticmethod
    def immediate_winning_moves(
        state: TicTacToeState,
        player: Mark,
    ) -> list[Move]:
        """Return immediate winning moves for ``player``

        Temporarily make ``player`` the next player when checking threats
        """

        player_state = replace(state, player_to_move=player)
        winning_moves: list[Move] = []

        for move in state.legal_moves():
            next_state = player_state.apply_move(move)
            if next_state.winner() == player:
                winning_moves.append(move)

        return winning_moves

    def centre_score(
        self,
        state: TicTacToeState,
        move: Move,
    ) -> float:
        """Score a move by its squared distance from the board centre."""

        centre_row = (state.rows - 1) / 2.0
        centre_col = (state.cols - 1) / 2.0
        squared_distance = (
            (move.row - centre_row) ** 2
            + (move.col - centre_col) ** 2
        )
        return self.centre_weight * -squared_distance

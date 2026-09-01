from __future__ import annotations

from random import Random

from src.games.tic_tac_toe import Move, TicTacToeState


class RandomAgent:
    """Baseline agent that selects uniformly from legal moves."""

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Return a random legal move using the supplied generator."""

        legal_moves = state.legal_moves()

        if not legal_moves:
            raise ValueError(
                "RandomAgent cannot choose a move from a terminal state."
            )

        return rng.choice(legal_moves)

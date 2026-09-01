from __future__ import annotations

from random import Random
from typing import Protocol

from src.games.tic_tac_toe import Move, TicTacToeState


class Agent(Protocol):
    """Common move-selection interface for Tic-Tac-Toe agents."""

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Choose and return one legal move for the given state."""
        ...

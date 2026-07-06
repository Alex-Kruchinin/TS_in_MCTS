from __future__ import annotations

from random import Random
from typing import Protocol

from src.games.tic_tac_toe import Move, TicTacToeState


class Agent(Protocol):
    """
    Common interface required from every Tic-Tac-Toe agent.

    An Agent is any object that provides a choose_move() method with
    this signature. Random, tactical, UCT-MCTS, and Thompson Sampling
    MCTS agents can therefore all be used by the same MatchRunner.
    """

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Choose and return one legal move for the supplied state."""


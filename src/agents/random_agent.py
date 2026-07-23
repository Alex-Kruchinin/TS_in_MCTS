from __future__ import annotations

from random import Random

from src.games.tic_tac_toe import Move, TicTacToeState


class RandomAgent:
    """
    An agent that selects uniformly from all currently legal moves.

    This agent does not evaluate positions or look ahead. It is useful
    as the easiest opponent and as a baseline for evaluating whether a
    search algorithm performs better than arbitrary play.
    """

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """
        Return one randomly selected legal move.

        The random-number generator is supplied by MatchRunner rather
        than created inside the agent. This allows a match to be repeated
        exactly by running it again with the same random seed.
        """

        legal_moves = state.legal_moves()

        if not legal_moves:
            raise ValueError(
                "RandomAgent cannot choose a move from a terminal state."
            )

        return rng.choice(legal_moves)

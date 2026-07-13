from __future__ import annotations

from random import Random

import pytest

from src.agents.random_agent import RandomAgent
from src.games.tic_tac_toe import Mark, Move, TicTacToeState
from src.matches.match_runner import MatchRunner


class ScriptedAgent:
    """Test-only agent that returns moves from a predefined sequence."""

    def __init__(self, moves: list[Move]) -> None:
        self.moves = iter(moves)

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        return next(self.moves)


class IllegalAgent:
    """Test-only agent that deliberately returns an invalid move."""

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        return Move(-1, -1)


def test_match_runner_alternates_agents_until_x_wins() -> None:
    runner = MatchRunner()

    agent_x = ScriptedAgent(
        [Move(0, 0), Move(0, 1), Move(0, 2)]
    )
    agent_o = ScriptedAgent(
        [Move(1, 0), Move(1, 1)]
    )

    result = runner.play(agent_x=agent_x, agent_o=agent_o, seed=7)

    # assert means: This condition must be true. If it is false, the test fails.
    assert result.winner == Mark.X
    assert not result.is_draw
    assert result.number_of_moves == 5
    assert result.move_history == (
        Move(0, 0),
        Move(1, 0),
        Move(0, 1),
        Move(1, 1),
        Move(0, 2),
    )
    assert result.final_state.is_terminal()


def test_random_agents_complete_a_scalable_match() -> None:
    runner = MatchRunner(rows=5, cols=5, win_length=4)

    result = runner.play(
        agent_x=RandomAgent(),
        agent_o=RandomAgent(),
        seed=42,
    )

    assert result.final_state.is_terminal()
    assert result.number_of_moves == result.final_state.moves_played
    assert 1 <= result.number_of_moves <= 25
    assert result.winner in (Mark.X, Mark.O, None)


def test_same_seed_reproduces_the_same_random_match() -> None:
    runner = MatchRunner(rows=8, cols=4, win_length=4)

    first_result = runner.play(
        RandomAgent(),
        RandomAgent(),
        seed=123,
    )
    repeated_result = runner.play(
        RandomAgent(),
        RandomAgent(),
        seed=123,
    )

    assert first_result.move_history == repeated_result.move_history
    assert first_result.winner == repeated_result.winner


def test_match_runner_rejects_an_illegal_agent_move() -> None:
    runner = MatchRunner()

    with pytest.raises(ValueError, match="illegal move"):
        runner.play(
            agent_x=IllegalAgent(),
            agent_o=RandomAgent(),
            seed=1,
        )

from random import Random

import pytest

from src.agents.random_agent import RandomAgent
from src.games.tic_tac_toe import Move, TicTacToeState


def test_random_agent_returns_a_legal_move() -> None:
    state = TicTacToeState.new(rows=5, cols=5, win_length=4)
    state = state.apply_move(Move(2, 2))

    agent = RandomAgent()
    move = agent.choose_move(state, Random(42))

    assert state.is_legal_move(move)
    assert move != Move(2, 2)


def test_random_agent_is_reproducible_with_same_seed() -> None:
    state = TicTacToeState.new(rows=8, cols=4, win_length=4)
    agent = RandomAgent()

    first_move = agent.choose_move(state, Random(123))
    repeated_move = agent.choose_move(state, Random(123))

    assert first_move == repeated_move


def test_random_agent_rejects_terminal_state() -> None:
    state = TicTacToeState.new()

    for move in (
        Move(0, 0),
        Move(1, 0),
        Move(0, 1),
        Move(1, 1),
        Move(0, 2),
    ):
        state = state.apply_move(move)

    assert state.is_terminal()

    with pytest.raises(ValueError):
        RandomAgent().choose_move(state, Random(1))

from __future__ import annotations

from random import Random

import pytest

from src.agents.tactical_agent import TacticalAgent
from src.games.tic_tac_toe import Mark, Move, TicTacToeState
from src.matches.match_runner import MatchRunner


def test_tactical_agent_takes_immediate_win() -> None:
    agent = TacticalAgent()
    rng = Random(0)

    state = TicTacToeState.new()
    state = state.apply_move(Move(0, 0))  # X
    state = state.apply_move(Move(1, 0))  # O
    state = state.apply_move(Move(0, 1))  # X
    state = state.apply_move(Move(1, 1))  # O

    # X can win immediately by completing the top row
    assert state.player_to_move == Mark.X
    assert agent.choose_move(state, rng) == Move(0, 2)


def test_tactical_agent_blocks_immediate_opponent_win() -> None:
    agent = TacticalAgent()
    rng = Random(0)

    state = TicTacToeState.new()
    state = state.apply_move(Move(1, 0))  # X
    state = state.apply_move(Move(0, 0))  # O
    state = state.apply_move(Move(2, 1))  # X
    state = state.apply_move(Move(0, 1))  # O

    # O threatens Move(0, 2), so X should block it
    assert state.player_to_move == Mark.X
    assert agent.choose_move(state, rng) == Move(0, 2)


def test_tactical_agent_prefers_centre_on_empty_standard_board() -> None:
    agent = TacticalAgent()
    rng = Random(0)

    state = TicTacToeState.new()

    assert agent.choose_move(state, rng) == Move(1, 1)


def test_tactical_agent_finds_immediate_win_on_larger_board() -> None:
    agent = TacticalAgent()
    rng = Random(0)

    state = TicTacToeState.new(
        rows=5,
        cols=5,
        win_length=4,
    )

    state = state.apply_move(Move(2, 0))  # X
    state = state.apply_move(Move(0, 0))  # O
    state = state.apply_move(Move(2, 1))  # X
    state = state.apply_move(Move(0, 1))  # O
    state = state.apply_move(Move(2, 2))  # X
    state = state.apply_move(Move(1, 0))  # O

    # X can complete four in a row with Move(2, 3)
    assert state.player_to_move == Mark.X
    assert agent.choose_move(state, rng) == Move(2, 3)


def test_tactical_agent_returns_legal_move_on_rectangular_board() -> None:
    agent = TacticalAgent()
    rng = Random(4)

    state = TicTacToeState.new(
        rows=8,
        cols=4,
        win_length=4,
    )

    move = agent.choose_move(state, rng)

    assert state.is_legal_move(move)


def test_tactical_agent_rejects_terminal_state() -> None:
    agent = TacticalAgent()
    rng = Random(0)

    state = TicTacToeState.new()
    state = state.apply_move(Move(0, 0))  # X
    state = state.apply_move(Move(1, 0))  # O
    state = state.apply_move(Move(0, 1))  # X
    state = state.apply_move(Move(1, 1))  # O
    state = state.apply_move(Move(0, 2))  # X wins

    with pytest.raises(ValueError):
        agent.choose_move(state, rng)


def test_tactical_agent_can_complete_match_against_random_agent() -> None:
    from src.agents.random_agent import RandomAgent

    runner = MatchRunner(
        rows=3,
        cols=3,
        win_length=3,
    )

    result = runner.play(
        agent_x=TacticalAgent(),
        agent_o=RandomAgent(),
        seed=12,
        record_trace=True,
    )

    assert result.final_state.is_terminal()
    assert result.number_of_moves <= 9
    assert result.trace is not None
    assert result.trace.agent_x_name == "TacticalAgent"

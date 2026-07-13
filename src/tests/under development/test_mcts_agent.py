from random import Random

import pytest

from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.games.tic_tac_toe import Mark, Move, TicTacToeState
from src.matches.match_runner import MatchRunner


def build_state(*moves: Move) -> TicTacToeState:
    state = TicTacToeState.new()

    for move in moves:
        state = state.apply_move(move)

    return state


def test_mcts_returns_a_legal_move() -> None:
    state = TicTacToeState.new(
        rows=5,
        cols=5,
        win_length=4,
    )
    agent = MCTSAgent(simulations=50)

    move = agent.choose_move(state, Random(10))

    assert state.is_legal_move(move)


def test_search_performs_requested_number_of_root_visits() -> None:
    state = TicTacToeState.new()
    agent = MCTSAgent(simulations=75)

    root = agent.search(state, Random(3))

    assert root.visits == 75
    assert sum(child.visits for child in root.children.values()) == 75


def test_mcts_selects_immediate_winning_move() -> None:
    # X X .
    # O O .
    # . . .
    # X can win immediately at (0, 2).
    state = build_state(
        Move(0, 0),
        Move(1, 0),
        Move(0, 1),
        Move(1, 1),
    )

    agent = MCTSAgent(simulations=500)
    move = agent.choose_move(state, Random(7))

    assert move == Move(0, 2)


def test_mcts_can_complete_match_against_random_agent() -> None:
    runner = MatchRunner()

    result = runner.play(
        agent_x=MCTSAgent(simulations=100),
        agent_o=RandomAgent(),
        seed=11,
    )

    assert result.final_state.is_terminal()
    assert result.number_of_moves <= 9


def test_mcts_rejects_terminal_state() -> None:
    terminal_state = build_state(
        Move(0, 0),
        Move(1, 0),
        Move(0, 1),
        Move(1, 1),
        Move(0, 2),
    )

    with pytest.raises(ValueError):
        MCTSAgent(simulations=10).choose_move(
            terminal_state,
            Random(1),
        )


def test_mcts_configuration_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        MCTSAgent(simulations=0)

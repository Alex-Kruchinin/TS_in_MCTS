import pytest

from agents import RandomAgent
from game import OthelloState


def test_random_agent_always_selects_legal_action() -> None:
    state = OthelloState.new()
    agent = RandomAgent(seed=10)

    assert agent.choose_action(state) in state.legal_actions()


def test_same_seed_produces_same_first_action() -> None:
    state = OthelloState.new()

    first = RandomAgent(seed=123).choose_action(state)
    second = RandomAgent(seed=123).choose_action(state)

    assert first == second


def test_random_agent_rejects_terminal_state() -> None:
    state = OthelloState.from_strings(
        [
            "BBBB",
            "BBBB",
            "BBBW",
            "BBWW",
        ]
    )

    with pytest.raises(ValueError, match="terminal"):
        RandomAgent(seed=1).choose_action(state)

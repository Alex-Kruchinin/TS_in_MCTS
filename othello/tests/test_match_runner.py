import pytest

from agents import RandomAgent
from game import Disc, Move, OthelloState, PASS
from matches import run_match


def test_random_match_reaches_valid_terminal_state() -> None:
    initial_state = OthelloState.new(rows=4, cols=4)

    result = run_match(
        RandomAgent(seed=1, name="Black"),
        RandomAgent(seed=2, name="White"),
        initial_state=initial_state,
    )

    assert result.final_state.is_terminal()
    assert result.final_state.legal_actions() == ()
    assert result.black_count + result.white_count <= 16
    assert result.black_count + result.white_count > 4
    assert result.action_count == len(result.actions)
    assert result.winner in (Disc.BLACK, Disc.WHITE, None)


def test_seeded_random_match_is_reproducible() -> None:
    def play():
        return run_match(
            RandomAgent(seed=100),
            RandomAgent(seed=200),
            initial_state=OthelloState.new(rows=6, cols=6),
        )

    first = play()
    second = play()

    assert first.actions == second.actions
    assert first.final_state == second.final_state
    assert first.winner == second.winner


def test_runner_supports_forced_pass() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    result = run_match(
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        initial_state=state,
    )

    assert result.actions[0] == PASS
    assert result.actions[1] == Move(3, 2)
    assert result.pass_count == 1
    assert result.final_state.is_terminal()


class IllegalAgent:
    @property
    def name(self) -> str:
        return "IllegalAgent"

    def choose_action(self, state: OthelloState) -> Move:
        return Move(0, 0)


def test_runner_rejects_illegal_agent_action() -> None:
    with pytest.raises(ValueError, match="illegal action"):
        run_match(
            IllegalAgent(),
            RandomAgent(seed=2),
            initial_state=OthelloState.new(),
        )

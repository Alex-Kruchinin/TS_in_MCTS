import pytest

from agents import TacticalAgent
from game import Disc, Move, OthelloState, PASS
from matches import run_match


def test_tactical_agent_takes_available_corner() -> None:
    state = OthelloState.from_strings(
        [
            ".WB.B.",
            "..BB..",
            ".BBW..",
            "BWWW..",
            "......",
            "......",
        ],
        current_player=Disc.BLACK,
    )
    agent = TacticalAgent(
        search_depth=2,
        exact_endgame_empty=0,
        seed=1,
    )

    assert Move(0, 0) in state.legal_actions()
    assert agent.choose_action(state) == Move(0, 0)


def test_tactical_agent_avoids_x_square_near_empty_corner() -> None:
    state = OthelloState.from_strings(
        [
            "......",
            "..B...",
            "..BB..",
            "..BW..",
            "......",
            "......",
        ],
        current_player=Disc.WHITE,
    )
    agent = TacticalAgent(
        search_depth=2,
        exact_endgame_empty=0,
        seed=1,
    )

    dangerous_x_square = Move(1, 1)
    assert dangerous_x_square in state.legal_actions()
    assert agent.choose_action(state) != dangerous_x_square


def test_tactical_agent_returns_forced_pass() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )
    agent = TacticalAgent(seed=1)

    assert state.legal_actions() == (PASS,)
    assert agent.choose_action(state) == PASS


def test_tactical_agent_records_action_scores() -> None:
    state = OthelloState.new(rows=4, cols=4)
    agent = TacticalAgent(
        search_depth=1,
        exact_endgame_empty=0,
        seed=1,
    )

    chosen = agent.choose_action(state)

    assert chosen in state.legal_actions()
    assert set(agent.last_action_scores) == set(state.legal_actions())
    assert all(
        isinstance(score, float)
        for score in agent.last_action_scores.values()
    )


def test_tactical_agent_works_on_rectangular_board() -> None:
    state = OthelloState.new(rows=6, cols=10)
    agent = TacticalAgent(
        search_depth=1,
        exact_endgame_empty=0,
        seed=5,
    )

    assert agent.choose_action(state) in state.legal_actions()


def test_tactical_agent_match_against_mcts_reaches_terminal_state() -> None:
    from agents import MCTSAgent

    result = run_match(
        TacticalAgent(
            search_depth=2,
            exact_endgame_empty=6,
            seed=10,
        ),
        MCTSAgent(simulations=20, seed=20),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()
    assert result.black_count + result.white_count <= 16


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("search_depth", 0),
        ("exact_endgame_empty", -1),
    ],
)
def test_tactical_agent_rejects_invalid_configuration(
    keyword: str,
    value: int,
) -> None:
    with pytest.raises(ValueError):
        TacticalAgent(**{keyword: value})

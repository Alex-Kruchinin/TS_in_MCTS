import pytest

from agents import EnhancedMCTSAgent, RandomAgent, TacticalAgent
from game import Disc, Move, OthelloState, PASS
from matches import run_match
from mcts import OthelloHeuristicPolicy


def test_heuristic_policy_prefers_available_corner() -> None:
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
    policy = OthelloHeuristicPolicy()

    corner = Move(0, 0)
    scores = {
        action: policy.action_score(state, action)
        for action in state.legal_actions()
    }

    assert corner in scores
    assert scores[corner] == max(scores.values())


def test_heuristic_policy_penalises_x_square_near_empty_corner() -> None:
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
    policy = OthelloHeuristicPolicy()
    dangerous = Move(1, 1)

    assert dangerous in state.legal_actions()
    dangerous_score = policy.action_score(state, dangerous)
    assert any(
        policy.action_score(state, action) > dangerous_score
        for action in state.legal_actions()
        if action != dangerous
    )



def test_enhanced_mcts_takes_available_corner() -> None:
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
    agent = EnhancedMCTSAgent(simulations=10, seed=1)

    assert agent.choose_action(state) == Move(0, 0)

def test_enhanced_mcts_returns_legal_action() -> None:
    state = OthelloState.new(rows=4, cols=4)
    agent = EnhancedMCTSAgent(simulations=20, seed=10)

    action = agent.choose_action(state)

    assert action in state.legal_actions()
    assert agent.last_root is not None
    assert agent.last_root.visits == 20


def test_seeded_enhanced_mcts_is_reproducible() -> None:
    state = OthelloState.new(rows=4, cols=4)

    first = EnhancedMCTSAgent(simulations=30, seed=99).choose_action(state)
    second = EnhancedMCTSAgent(simulations=30, seed=99).choose_action(state)

    assert first == second


def test_enhanced_mcts_handles_forced_pass() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    action = EnhancedMCTSAgent(simulations=10, seed=3).choose_action(state)

    assert action == PASS


def test_enhanced_mcts_can_complete_match_against_random() -> None:
    result = run_match(
        EnhancedMCTSAgent(simulations=30, seed=1),
        RandomAgent(seed=2),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()
    assert result.action_count > 0


def test_enhanced_mcts_can_complete_match_against_tactical() -> None:
    result = run_match(
        EnhancedMCTSAgent(simulations=20, seed=10),
        TacticalAgent(search_depth=1, exact_endgame_empty=4, seed=20),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("simulations", 0),
        ("exploration_constant", -0.1),
        ("progressive_bias_weight", -0.1),
        ("rollout_epsilon", -0.1),
        ("rollout_epsilon", 1.1),
        ("expansion_epsilon", -0.1),
        ("expansion_epsilon", 1.1),
        ("rollout_depth_limit", 0),
    ],
)
def test_enhanced_mcts_rejects_invalid_configuration(
    keyword: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        EnhancedMCTSAgent(**{keyword: value})


def test_enhanced_mcts_feature_switches_are_exposed() -> None:
    agent = EnhancedMCTSAgent(
        simulations=5,
        guided_expansion=False,
        heuristic_rollouts=False,
        root_safety=False,
        root_lookahead=False,
        heuristic_cutoff=False,
        seed=1,
    )

    assert agent.guided_expansion is False
    assert agent.heuristic_rollouts is False
    assert agent.root_safety is False
    assert agent.root_lookahead is False
    assert agent.heuristic_cutoff is False


def test_enhanced_mcts_without_heuristic_cutoff_completes_search() -> None:
    state = OthelloState.new(rows=4, cols=4)
    agent = EnhancedMCTSAgent(
        simulations=5,
        heuristic_cutoff=False,
        seed=7,
    )

    assert agent.choose_action(state) in state.legal_actions()

import random

import pytest

from agents import (
    EnhancedMCTSAgent,
    EnhancedThompsonMCTSAgent,
    RandomAgent,
    TacticalAgent,
    ThompsonMCTSAgent,
)
from game import Disc, Move, OthelloState, PASS
from matches import run_match
from mcts import ThompsonMCTSNode


class PosteriorMeanRng:
    """Small deterministic RNG stub for testing child selection."""

    @staticmethod
    def betavariate(alpha: float, beta: float) -> float:
        return alpha / (alpha + beta)

    @staticmethod
    def choice(values):
        return values[0]


def black_winning_terminal_state() -> OthelloState:
    return OthelloState.from_strings(
        [
            "BBBB",
            "BBBB",
            "BBBW",
            "BBWW",
        ],
        current_player=Disc.WHITE,
    )


def drawn_terminal_state() -> OthelloState:
    return OthelloState.from_strings(
        [
            "BBWW",
            "BBWW",
            "WWBB",
            "WWBB",
        ],
        current_player=Disc.BLACK,
    )


def test_thompson_node_starts_with_uniform_beta_prior() -> None:
    node = ThompsonMCTSNode(OthelloState.new(rows=4, cols=4))

    assert node.alpha == 1.0
    assert node.beta == 1.0
    assert node.posterior_mean == 0.5


def test_win_updates_success_evidence_for_player_just_moved() -> None:
    root = ThompsonMCTSNode(OthelloState.new(rows=4, cols=4))
    child = root.expand(random.Random(1))

    child.backpropagate(black_winning_terminal_state())

    assert child.player_just_moved is Disc.BLACK
    assert child.visits == 1
    assert child.total_value == 1.0
    assert child.alpha == 2.0
    assert child.beta == 1.0

    # The root's bookkeeping perspective is White, so the same outcome is a
    # failure there. Root posterior values are not used for the final move.
    assert root.visits == 1
    assert root.alpha == 1.0
    assert root.beta == 2.0


def test_draw_adds_half_success_and_half_failure_evidence() -> None:
    root = ThompsonMCTSNode(OthelloState.new(rows=4, cols=4))
    child = root.expand(random.Random(1))

    child.backpropagate(drawn_terminal_state())

    assert child.total_value == 0.5
    assert child.alpha == 1.5
    assert child.beta == 1.5
    assert child.posterior_mean == 0.5


def test_thompson_selection_prefers_larger_posterior_sample() -> None:
    root = ThompsonMCTSNode(OthelloState.new(rows=4, cols=4))
    first = root.expand(random.Random(1))
    second = root.expand(random.Random(2))

    first.visits = 10
    first.total_value = 8.0
    second.visits = 10
    second.total_value = 2.0

    assert root.sample_child(PosteriorMeanRng()) is first


def test_classical_thompson_returns_legal_action() -> None:
    state = OthelloState.new(rows=4, cols=4)
    agent = ThompsonMCTSAgent(simulations=20, seed=10)

    action = agent.choose_action(state)

    assert action in state.legal_actions()
    assert agent.last_root is not None
    assert agent.last_root.visits == 20
    assert sum(child.visits for child in agent.last_root.children) == 20


def test_seeded_classical_thompson_is_reproducible() -> None:
    state = OthelloState.new(rows=4, cols=4)

    first = ThompsonMCTSAgent(simulations=30, seed=99).choose_action(state)
    second = ThompsonMCTSAgent(simulations=30, seed=99).choose_action(state)

    assert first == second


def test_classical_thompson_handles_forced_pass() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    action = ThompsonMCTSAgent(simulations=10, seed=3).choose_action(state)

    assert action == PASS


def test_classical_thompson_completes_match_against_random() -> None:
    result = run_match(
        ThompsonMCTSAgent(simulations=30, seed=1),
        RandomAgent(seed=2),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()
    assert result.action_count > 0


def test_enhanced_thompson_returns_legal_action() -> None:
    state = OthelloState.new(rows=4, cols=4)
    agent = EnhancedThompsonMCTSAgent(simulations=20, seed=10)

    action = agent.choose_action(state)

    assert action in state.legal_actions()
    assert agent.last_root is not None
    assert agent.last_root.visits == 20


def test_seeded_enhanced_thompson_is_reproducible() -> None:
    state = OthelloState.new(rows=4, cols=4)

    first = EnhancedThompsonMCTSAgent(
        simulations=30,
        seed=99,
    ).choose_action(state)
    second = EnhancedThompsonMCTSAgent(
        simulations=30,
        seed=99,
    ).choose_action(state)

    assert first == second


def test_enhanced_thompson_takes_available_corner() -> None:
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

    action = EnhancedThompsonMCTSAgent(
        simulations=10,
        seed=1,
    ).choose_action(state)

    assert action == Move(0, 0)


def test_enhanced_thompson_handles_forced_pass() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    action = EnhancedThompsonMCTSAgent(
        simulations=10,
        seed=3,
    ).choose_action(state)

    assert action == PASS


def test_enhanced_thompson_completes_match_against_tactical() -> None:
    result = run_match(
        EnhancedThompsonMCTSAgent(simulations=15, seed=10),
        TacticalAgent(search_depth=1, exact_endgame_empty=4, seed=20),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()


def test_enhanced_thompson_and_enhanced_uct_are_compatible() -> None:
    result = run_match(
        EnhancedThompsonMCTSAgent(simulations=12, seed=1),
        EnhancedMCTSAgent(
            simulations=12,
            progressive_bias_weight=0.0,
            seed=2,
        ),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()


def test_enhanced_thompson_feature_switches_are_exposed() -> None:
    agent = EnhancedThompsonMCTSAgent(
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


@pytest.mark.parametrize(
    ("agent_type", "keyword", "value"),
    [
        (ThompsonMCTSAgent, "simulations", 0),
        (ThompsonMCTSAgent, "alpha_prior", 0.0),
        (ThompsonMCTSAgent, "beta_prior", -1.0),
        (ThompsonMCTSAgent, "alpha_prior", float("inf")),
        (ThompsonMCTSAgent, "beta_prior", float("nan")),
        (EnhancedThompsonMCTSAgent, "simulations", 0),
        (EnhancedThompsonMCTSAgent, "alpha_prior", 0.0),
        (EnhancedThompsonMCTSAgent, "beta_prior", -1.0),
        (EnhancedThompsonMCTSAgent, "alpha_prior", float("inf")),
        (EnhancedThompsonMCTSAgent, "beta_prior", float("nan")),
        (EnhancedThompsonMCTSAgent, "rollout_epsilon", -0.1),
        (EnhancedThompsonMCTSAgent, "rollout_epsilon", 1.1),
        (EnhancedThompsonMCTSAgent, "expansion_epsilon", -0.1),
        (EnhancedThompsonMCTSAgent, "expansion_epsilon", 1.1),
        (EnhancedThompsonMCTSAgent, "rollout_depth_limit", 0),
    ],
)
def test_thompson_agents_reject_invalid_configuration(
    agent_type,
    keyword: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        agent_type(**{keyword: value})

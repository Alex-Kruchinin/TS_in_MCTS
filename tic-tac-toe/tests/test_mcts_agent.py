from dataclasses import replace
from random import Random

import pytest

from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.agents.selection_policies import ThompsonSamplingSelectionPolicy
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



def test_baseline_thompson_search_performs_requested_root_visits() -> None:
    state = TicTacToeState.new()
    agent = MCTSAgent.baseline_thompson(simulations=60)

    root = agent.search(state, Random(4))

    assert root.visits == 60
    assert isinstance(agent.selection_policy, ThompsonSamplingSelectionPolicy)


def test_enhanced_thompson_returns_legal_move_on_larger_board() -> None:
    state = TicTacToeState.new(
        rows=7,
        cols=7,
        win_length=4,
    )
    agent = MCTSAgent.enhanced_thompson(simulations=30)

    move = agent.choose_move(state, Random(6))

    assert state.is_legal_move(move)
    assert agent.rollout_policy == "internal_heuristic"
    assert agent.rollout_policy_name() == "InternalHeuristicRollout"
    assert agent.use_tactical_guard
    assert agent.use_heuristic_expansion


def test_mcts_agent_does_not_embed_tactical_agent_for_rollout() -> None:
    agent = MCTSAgent.enhanced_uct(simulations=10)

    assert not hasattr(agent, "rollout_agent")
    assert not hasattr(agent, "heuristic_agent")
    assert agent.rollout_policy == "internal_heuristic"


def test_enhanced_agents_enable_internal_fork_detection() -> None:
    enhanced_uct = MCTSAgent.enhanced_uct(simulations=10)
    enhanced_thompson = MCTSAgent.enhanced_thompson(simulations=10)
    baseline_uct = MCTSAgent.baseline_uct(simulations=10)

    assert enhanced_uct.use_fork_detection
    assert enhanced_thompson.use_fork_detection
    assert not baseline_uct.use_fork_detection


def test_internal_fork_detector_finds_double_threat_move() -> None:
    # X . .
    # O . .
    # . O X
    #
    # X to move. Move(0, 2) creates two immediate winning threats:
    #   Move(0, 1) completes the top row.
    #   Move(1, 2) completes the right column.
    state = build_state(
        Move(0, 0),
        Move(1, 0),
        Move(2, 2),
        Move(2, 1),
    )
    agent = MCTSAgent.enhanced_uct(simulations=10)

    fork_moves = agent._fork_creating_moves(state, Mark.X)
    threat_count = agent._future_winning_threat_count_after_move(
        state=state,
        move=Move(0, 2),
        player=Mark.X,
    )

    assert Move(0, 2) in fork_moves
    assert threat_count >= agent.fork_threshold


def test_tactical_guard_can_choose_fork_block() -> None:
    # X X O
    # O . .
    # . . .
    #
    # O to move. X has no immediate win, but if O ignores the centre then X
    # can play Move(1, 1) and create multiple future winning threats. The
    # enhanced guard should therefore occupy the fork square.
    state = build_state(
        Move(0, 0),
        Move(0, 2),
        Move(0, 1),
        Move(1, 0),
    )
    state = replace(state, player_to_move=Mark.O)
    agent = MCTSAgent.enhanced_thompson(simulations=10)

    assert not agent._immediate_winning_moves(state, Mark.X)
    opponent_fork_moves = agent._fork_creating_moves(state, Mark.X)
    move = agent._tactical_guard_move(state, Random(1))

    assert opponent_fork_moves == [Move(1, 1)]
    assert move == Move(1, 1)


def _slow_reference_immediate_wins(
    state: TicTacToeState,
    player: Mark,
) -> list[Move]:
    """Reference version matching the pre-optimisation implementation."""

    player_state = replace(state, player_to_move=player)
    result: list[Move] = []
    for move in state.legal_moves():
        next_state = player_state.apply_move(move)
        if next_state.winner() == player:
            result.append(move)
    return result


def _slow_reference_future_threat_count(
    state: TicTacToeState,
    move: Move,
    player: Mark,
) -> int:
    if not state.is_legal_move(move):
        return 0

    player_state = replace(state, player_to_move=player)
    next_state = player_state.apply_move(move)
    if next_state.winner() == player:
        return 0

    return len(_slow_reference_immediate_wins(next_state, player))


def test_optimised_immediate_win_scan_matches_reference() -> None:
    state = TicTacToeState.new(rows=7, cols=7, win_length=4)
    for move in (
        Move(3, 3), Move(0, 0), Move(3, 2), Move(1, 1),
        Move(3, 4), Move(2, 2),
    ):
        state = state.apply_move(move)

    agent = MCTSAgent.enhanced_uct(simulations=10)
    for player in (Mark.X, Mark.O):
        assert agent._immediate_winning_moves(state, player) == (
            _slow_reference_immediate_wins(state, player)
        )


def test_optimised_fork_threat_count_matches_reference() -> None:
    state = TicTacToeState.new(rows=7, cols=7, win_length=4)
    for move in (
        Move(3, 3), Move(0, 0), Move(3, 2), Move(1, 1),
        Move(3, 4), Move(2, 2),
    ):
        state = state.apply_move(move)

    agent = MCTSAgent.enhanced_uct(simulations=10)
    for player in (Mark.X, Mark.O):
        for move in state.legal_moves():
            assert agent._future_winning_threat_count_after_move(
                state, move, player
            ) == _slow_reference_future_threat_count(
                state, move, player
            )

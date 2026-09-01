from random import Random

from src.agents.mcts_node import MCTSNode
from src.agents.selection_policies import (
    ProgressiveBiasUCTSelectionPolicy,
    ThompsonSamplingSelectionPolicy,
    UCTSelectionPolicy,
)
from src.games.tic_tac_toe import Move, TicTacToeState


def test_uct_prefers_high_value_when_exploration_is_zero() -> None:
    root = MCTSNode(TicTacToeState.new())

    child_a = MCTSNode(
        state=root.state.apply_move(Move(0, 0)),
        parent=root,
        move=Move(0, 0),
    )
    child_b = MCTSNode(
        state=root.state.apply_move(Move(0, 1)),
        parent=root,
        move=Move(0, 1),
    )

    root.children = {
        child_a.move: child_a,
        child_b.move: child_b,
    }
    root.untried_moves = []
    root.visits = 20

    child_a.visits = 10
    child_a.value_sum = 8.0

    child_b.visits = 10
    child_b.value_sum = 3.0

    policy = UCTSelectionPolicy(exploration_constant=0.0)
    selected = policy.select_child(root, Random(1))

    assert selected is child_a


def test_uct_explores_an_unvisited_child() -> None:
    root = MCTSNode(TicTacToeState.new())

    visited_child = MCTSNode(
        state=root.state.apply_move(Move(0, 0)),
        parent=root,
        move=Move(0, 0),
    )
    unvisited_child = MCTSNode(
        state=root.state.apply_move(Move(0, 1)),
        parent=root,
        move=Move(0, 1),
    )

    root.children = {
        visited_child.move: visited_child,
        unvisited_child.move: unvisited_child,
    }
    root.untried_moves = []
    root.visits = 10

    visited_child.visits = 10
    visited_child.value_sum = 10.0

    policy = UCTSelectionPolicy()
    selected = policy.select_child(root, Random(1))

    assert selected is unvisited_child



def test_thompson_sampling_prefers_child_with_strong_posterior() -> None:
    root = MCTSNode(TicTacToeState.new())

    weak_child = MCTSNode(
        state=root.state.apply_move(Move(0, 0)),
        parent=root,
        move=Move(0, 0),
    )
    strong_child = MCTSNode(
        state=root.state.apply_move(Move(1, 1)),
        parent=root,
        move=Move(1, 1),
    )

    root.children = {
        weak_child.move: weak_child,
        strong_child.move: strong_child,
    }
    root.untried_moves = []

    weak_child.alpha = 1.0
    weak_child.beta = 100.0
    strong_child.alpha = 100.0
    strong_child.beta = 1.0

    policy = ThompsonSamplingSelectionPolicy()
    selected = policy.select_child(root, Random(5))

    assert selected is strong_child


def test_progressive_bias_can_use_heuristic_when_values_are_tied() -> None:
    root = MCTSNode(TicTacToeState.new())

    low_heuristic_child = MCTSNode(
        state=root.state.apply_move(Move(0, 0)),
        parent=root,
        move=Move(0, 0),
        heuristic_value=0.0,
    )
    high_heuristic_child = MCTSNode(
        state=root.state.apply_move(Move(1, 1)),
        parent=root,
        move=Move(1, 1),
        heuristic_value=1.0,
    )

    root.children = {
        low_heuristic_child.move: low_heuristic_child,
        high_heuristic_child.move: high_heuristic_child,
    }
    root.untried_moves = []
    root.visits = 20

    for child in root.children.values():
        child.visits = 10
        child.value_sum = 5.0

    policy = ProgressiveBiasUCTSelectionPolicy(
        exploration_constant=0.0,
        heuristic_weight=1.0,
    )
    selected = policy.select_child(root, Random(1))

    assert selected is high_heuristic_child

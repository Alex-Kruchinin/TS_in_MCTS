from random import Random

from src.agents.mcts_node import MCTSNode
from src.agents.selection_policies import UCTSelectionPolicy
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

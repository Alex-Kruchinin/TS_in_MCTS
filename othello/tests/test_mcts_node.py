import math
import random

import pytest

from game import Disc, OthelloState
from mcts import MCTSNode


def test_root_node_starts_with_all_legal_actions_untried() -> None:
    state = OthelloState.new(rows=4, cols=4)
    node = MCTSNode(state)

    assert node.parent is None
    assert node.action is None
    assert node.children == []
    assert set(node.untried_actions) == set(state.legal_actions())
    assert node.visits == 0
    assert node.mean_value == 0.0


def test_expand_creates_one_child_and_removes_one_untried_action() -> None:
    root = MCTSNode(OthelloState.new(rows=4, cols=4))
    initial_untried_count = len(root.untried_actions)

    child = root.expand(random.Random(1))

    assert child.parent is root
    assert child in root.children
    assert child.action is not None
    assert child.state == root.state.apply(child.action)
    assert len(root.untried_actions) == initial_untried_count - 1


def test_best_child_with_zero_exploration_uses_mean_value() -> None:
    root = MCTSNode(OthelloState.new(rows=4, cols=4))
    first = root.expand(random.Random(1))
    second = root.expand(random.Random(2))

    root.visits = 20
    first.visits = 10
    first.total_value = 8.0
    second.visits = 10
    second.total_value = 3.0

    selected = root.best_child(0.0, random.Random(3))

    assert selected is first


def test_unvisited_child_has_priority_in_uct_selection() -> None:
    root = MCTSNode(OthelloState.new(rows=4, cols=4))
    visited = root.expand(random.Random(1))
    unvisited = root.expand(random.Random(2))

    root.visits = 10
    visited.visits = 10
    visited.total_value = 10.0

    selected = root.best_child(math.sqrt(2.0), random.Random(3))

    assert selected is unvisited


def test_backpropagation_updates_node_from_player_just_moved_perspective() -> None:
    terminal = OthelloState.from_strings(
        [
            "BBBB",
            "BBBB",
            "BBBW",
            "BBWW",
        ],
        current_player=Disc.WHITE,
    )
    node = MCTSNode(terminal)

    node.backpropagate(terminal)

    # WHITE is to move in the stored terminal state, so BLACK moved last.
    assert node.player_just_moved is Disc.BLACK
    assert node.visits == 1
    assert node.total_value == 1.0
    assert node.mean_value == 1.0


def test_terminal_node_cannot_be_expanded() -> None:
    terminal = OthelloState.from_strings(
        [
            "BBBB",
            "BBBB",
            "BBBB",
            "BBBB",
        ],
        current_player=Disc.WHITE,
    )

    with pytest.raises(ValueError, match="terminal"):
        MCTSNode(terminal).expand(random.Random(1))

from random import Random

import pytest

from src.agents.mcts_node import MCTSNode
from src.games.tic_tac_toe import Mark, Move, TicTacToeState


def play_moves(*moves: Move) -> TicTacToeState:
    state = TicTacToeState.new()

    for move in moves:
        state = state.apply_move(move)

    return state


def test_new_root_contains_every_legal_move_as_untried() -> None:
    state = TicTacToeState.new()
    root = MCTSNode(state)

    assert len(root.untried_moves) == 9
    assert root.children == {}
    assert root.visits == 0
    assert root.value_sum == 0.0


def test_expand_creates_one_child_and_removes_one_untried_move() -> None:
    root = MCTSNode(TicTacToeState.new())

    child = root.expand(Random(4))

    assert len(root.children) == 1
    assert len(root.untried_moves) == 8
    assert child.parent is root
    assert child.move in root.children
    assert child.state.moves_played == 1
    assert child.state.player_to_move == Mark.O


def test_update_records_win_for_player_who_entered_node() -> None:
    root = MCTSNode(TicTacToeState.new())
    child = root.expand(Random(1))

    # Construct a terminal game in which X wins. Every root child represents
    # an X move because X is the player moving at the root.
    terminal_state = play_moves(
        Move(0, 0),
        Move(1, 0),
        Move(0, 1),
        Move(1, 1),
        Move(0, 2),
    )

    child.update(terminal_state)

    assert child.player_just_moved == Mark.X
    assert child.visits == 1
    assert child.value_sum == 1.0
    assert child.mean_value == 1.0


def test_terminal_node_cannot_be_expanded() -> None:
    terminal_state = play_moves(
        Move(0, 0),
        Move(1, 0),
        Move(0, 1),
        Move(1, 1),
        Move(0, 2),
    )

    node = MCTSNode(terminal_state)

    with pytest.raises(ValueError):
        node.expand(Random(1))

import pytest

from src.games.tic_tac_toe import (
    Mark,
    Move,
    TicTacToeState,
)


def test_new_standard_board_is_empty() -> None:
    state = TicTacToeState.new()

    assert state.rows == 3
    assert state.cols == 3
    assert state.win_length == 3
    assert state.player_to_move == Mark.X
    assert state.moves_played == 0
    assert len(state.legal_moves()) == 9


def test_move_is_applied_to_correct_square() -> None:
    state = TicTacToeState.new()

    next_state = state.apply_move(Move(1, 1))

    assert next_state.cell_at(Move(1, 1)) == Mark.X
    assert next_state.player_to_move == Mark.O
    assert next_state.moves_played == 1
    assert next_state.last_move == Move(1, 1)


def test_original_state_is_not_modified() -> None:
    state = TicTacToeState.new()

    next_state = state.apply_move(Move(1, 1))

    assert state.cell_at(Move(1, 1)) == Mark.EMPTY
    assert next_state.cell_at(Move(1, 1)) == Mark.X


def test_cannot_play_in_occupied_square() -> None:
    state = TicTacToeState.new()
    state = state.apply_move(Move(1, 1))

    with pytest.raises(ValueError):
        state.apply_move(Move(1, 1))


def test_move_outside_board_is_illegal() -> None:
    state = TicTacToeState.new()

    assert not state.is_legal_move(Move(-1, 0))
    assert not state.is_legal_move(Move(3, 0))


def test_scalable_board_has_correct_number_of_moves() -> None:
    state = TicTacToeState.new(
        rows=5,
        cols=5,
        win_length=4,
    )

    assert len(state.board) == 25
    assert len(state.legal_moves()) == 25
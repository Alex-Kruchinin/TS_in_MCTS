import pytest

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


def play_moves(
    state: TicTacToeState,
    moves: list[Move],
) -> TicTacToeState:
    """Apply a sequence of moves and return the resulting state."""

    for move in moves:
        state = state.apply_move(move)

    return state


def test_new_standard_board_is_empty() -> None:
    state = TicTacToeState.new()

    assert state.rows == 3
    assert state.cols == 3
    assert state.win_length == 3
    assert state.player_to_move == Mark.X
    assert state.moves_played == 0
    assert state.winner() is None
    assert not state.is_terminal()
    assert len(state.legal_moves()) == 9


def test_rectangular_board_uses_columns_for_indexing() -> None:
    state = TicTacToeState.new(rows=8, cols=4, win_length=4)

    assert len(state.board) == 32
    assert state.move_to_index(Move(2, 1)) == 9
    assert state.index_to_move(9) == Move(2, 1)


def test_apply_move_creates_a_new_state() -> None:
    original_state = TicTacToeState.new()
    next_state = original_state.apply_move(Move(1, 1))

    assert original_state.cell_at(Move(1, 1)) == Mark.EMPTY
    assert next_state.cell_at(Move(1, 1)) == Mark.X
    assert next_state.player_to_move == Mark.O
    assert next_state.last_move == Move(1, 1)
    assert next_state.moves_played == 1


def test_horizontal_win_on_5_by_5_board() -> None:
    state = TicTacToeState.new(rows=5, cols=5, win_length=4)
    state = play_moves(
        state,
        [
            Move(2, 0),  # X
            Move(0, 0),  # O
            Move(2, 1),  # X
            Move(0, 1),  # O
            Move(2, 2),  # X
            Move(0, 2),  # O
            Move(2, 3),  # X wins
        ],
    )

    assert state.winner() == Mark.X
    assert state.is_terminal()
    assert not state.is_draw()


def test_vertical_win_on_8_by_4_board() -> None:
    state = TicTacToeState.new(rows=8, cols=4, win_length=4)
    state = play_moves(
        state,
        [
            Move(0, 1),  # X
            Move(0, 0),  # O
            Move(1, 1),  # X
            Move(1, 0),  # O
            Move(2, 1),  # X
            Move(2, 0),  # O
            Move(3, 1),  # X wins
        ],
    )

    assert state.winner() == Mark.X


def test_down_right_diagonal_win() -> None:
    state = TicTacToeState.new(rows=5, cols=5, win_length=4)
    state = play_moves(
        state,
        [
            Move(0, 0),  # X
            Move(0, 1),  # O
            Move(1, 1),  # X
            Move(0, 2),  # O
            Move(2, 2),  # X
            Move(0, 3),  # O
            Move(3, 3),  # X wins
        ],
    )

    assert state.winner() == Mark.X


def test_down_left_diagonal_win() -> None:
    state = TicTacToeState.new(rows=5, cols=5, win_length=4)
    state = play_moves(
        state,
        [
            Move(0, 4),  # X
            Move(0, 0),  # O
            Move(1, 3),  # X
            Move(0, 1),  # O
            Move(2, 2),  # X
            Move(0, 2),  # O
            Move(3, 1),  # X wins
        ],
    )

    assert state.winner() == Mark.X


def test_draw_is_detected() -> None:
    state = TicTacToeState.new()
    state = play_moves(
        state,
        [
            Move(0, 0),  # X
            Move(0, 1),  # O
            Move(0, 2),  # X
            Move(1, 1),  # O
            Move(1, 0),  # X
            Move(1, 2),  # O
            Move(2, 1),  # X
            Move(2, 0),  # O
            Move(2, 2),  # X
        ],
    )

    assert state.winner() is None
    assert state.is_draw()
    assert state.is_terminal()
    assert state.legal_moves() == ()


def test_rewards_are_returned_from_requested_players_perspective() -> None:
    state = TicTacToeState.new()
    state = play_moves(
        state,
        [
            Move(0, 0),  # X
            Move(1, 0),  # O
            Move(0, 1),  # X
            Move(1, 1),  # O
            Move(0, 2),  # X wins
        ],
    )

    assert state.reward_for(Mark.X) == 1.0
    assert state.reward_for(Mark.O) == 0.0


def test_draw_gives_both_players_half_reward() -> None:
    state = TicTacToeState.new()
    state = play_moves(
        state,
        [
            Move(0, 0),
            Move(0, 1),
            Move(0, 2),
            Move(1, 1),
            Move(1, 0),
            Move(1, 2),
            Move(2, 1),
            Move(2, 0),
            Move(2, 2),
        ],
    )

    assert state.reward_for(Mark.X) == 0.5
    assert state.reward_for(Mark.O) == 0.5


def test_reward_cannot_be_requested_before_game_ends() -> None:
    state = TicTacToeState.new()

    with pytest.raises(ValueError):
        state.reward_for(Mark.X)


def test_move_cannot_be_played_after_a_win() -> None:
    state = TicTacToeState.new()
    state = play_moves(
        state,
        [
            Move(0, 0),
            Move(1, 0),
            Move(0, 1),
            Move(1, 1),
            Move(0, 2),  # X wins while empty cells remain
        ],
    )

    assert state.is_terminal()
    assert state.legal_moves() == ()
    assert not state.is_legal_move(Move(2, 2))

    with pytest.raises(ValueError):
        state.apply_move(Move(2, 2))

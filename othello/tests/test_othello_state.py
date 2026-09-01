import pytest

from game import Disc, Move, OthelloState, PASS


def test_new_standard_board_has_correct_centre_and_player() -> None:
    state = OthelloState.new()

    assert state.rows == 8
    assert state.cols == 8
    assert state.current_player is Disc.BLACK
    assert state.disc_at(3, 3) is Disc.WHITE
    assert state.disc_at(3, 4) is Disc.BLACK
    assert state.disc_at(4, 3) is Disc.BLACK
    assert state.disc_at(4, 4) is Disc.WHITE
    assert state.disc_count(Disc.BLACK) == 2
    assert state.disc_count(Disc.WHITE) == 2
    assert state.disc_count(Disc.EMPTY) == 60


def test_new_rectangular_board_is_supported() -> None:
    state = OthelloState.new(rows=6, cols=10)

    assert state.disc_at(2, 4) is Disc.WHITE
    assert state.disc_at(2, 5) is Disc.BLACK
    assert state.disc_at(3, 4) is Disc.BLACK
    assert state.disc_at(3, 5) is Disc.WHITE
    assert len(state.board) == 60


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (3, 4),
        (4, 3),
        (5, 6),
        (6, 5),
    ],
)
def test_invalid_board_dimensions_are_rejected(
    rows: int,
    cols: int,
) -> None:
    with pytest.raises(ValueError):
        OthelloState.new(rows=rows, cols=cols)


def test_initial_black_legal_moves_are_correct() -> None:
    state = OthelloState.new()

    assert set(state.legal_placements()) == {
        Move(2, 3),
        Move(3, 2),
        Move(4, 5),
        Move(5, 4),
    }


def test_applying_initial_move_places_and_flips_disc() -> None:
    state = OthelloState.new()

    next_state = state.apply(Move(2, 3))

    assert next_state.disc_at(2, 3) is Disc.BLACK
    assert next_state.disc_at(3, 3) is Disc.BLACK
    assert next_state.current_player is Disc.WHITE
    assert next_state.disc_count(Disc.BLACK) == 4
    assert next_state.disc_count(Disc.WHITE) == 1


def test_applying_move_does_not_mutate_original_state() -> None:
    state = OthelloState.new()
    original_board = state.board

    next_state = state.apply(Move(2, 3))

    assert state.board == original_board
    assert state.disc_at(2, 3) is Disc.EMPTY
    assert next_state is not state


def test_one_move_can_capture_in_multiple_directions() -> None:
    state = OthelloState.from_strings(
        [
            "B..B....",
            ".W.W....",
            "..WW....",
            "BWW.....",
            "........",
            "........",
            "........",
            "........",
        ],
        current_player=Disc.BLACK,
    )

    captured = set(state.captured_indices(Move(3, 3)))

    assert captured == {
        state.move_to_index(Move(1, 1)),
        state.move_to_index(Move(2, 2)),
        state.move_to_index(Move(1, 3)),
        state.move_to_index(Move(2, 3)),
        state.move_to_index(Move(3, 1)),
        state.move_to_index(Move(3, 2)),
    }

    next_state = state.apply(Move(3, 3))
    assert all(
        next_state.board[index] is Disc.BLACK
        for index in captured
    )


@pytest.mark.parametrize(
    "move",
    [
        Move(-1, 0),
        Move(8, 0),
        Move(3, 3),
        Move(0, 0),
    ],
)
def test_illegal_placements_are_rejected(move: Move) -> None:
    state = OthelloState.new()

    with pytest.raises(ValueError):
        state.apply(move)


def test_pass_is_returned_when_only_opponent_can_move() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    assert state.legal_placements(Disc.BLACK) == ()
    assert state.legal_placements(Disc.WHITE) == (Move(3, 2),)
    assert state.legal_actions() == (PASS,)


def test_forced_pass_changes_player_without_changing_board() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    next_state = state.apply(PASS)

    assert next_state.board == state.board
    assert next_state.current_player is Disc.WHITE


def test_pass_is_rejected_when_normal_move_exists() -> None:
    state = OthelloState.new()

    with pytest.raises(ValueError):
        state.apply(PASS)


def test_terminal_winner_and_result_are_based_on_disc_count() -> None:
    state = OthelloState.from_strings(
        [
            "BBBB",
            "BBBB",
            "BBBW",
            "BBWW",
        ],
        current_player=Disc.WHITE,
    )

    assert state.is_terminal()
    assert state.legal_actions() == ()
    assert state.winner() is Disc.BLACK
    assert state.result_for(Disc.BLACK) == 1.0
    assert state.result_for(Disc.WHITE) == 0.0


def test_terminal_draw_returns_half_reward() -> None:
    state = OthelloState.from_strings(
        [
            "BBWW",
            "BBWW",
            "WWBB",
            "WWBB",
        ],
        current_player=Disc.BLACK,
    )

    assert state.is_terminal()
    assert state.winner() is None
    assert state.result_for(Disc.BLACK) == 0.5
    assert state.result_for(Disc.WHITE) == 0.5


def test_winner_cannot_be_requested_before_terminal_state() -> None:
    state = OthelloState.new()

    with pytest.raises(ValueError):
        state.winner()

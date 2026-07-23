from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


@dataclass(frozen=True, slots=True)
class TacticalAgent:
    """
    A rule-based Tic-Tac-Toe agent for medium-difficulty testing.

    This agent is stronger than RandomAgent but cheaper and simpler than
    MCTS. It follows a priority list:
        1. play an immediate winning move;
        2. block the opponent's immediate winning move;
        3. otherwise score legal moves using line extension, blocking
           value, and centre preference;
        4. randomly break ties.

    It is tactical rather than perfect: it does not perform deep search and
    can still miss multi-move traps such as forks.
    """

    # These weights make immediate line-building more important than the
    # small positional centre preference.
    own_line_weight: float = 10.0
    opponent_line_weight: float = 8.0
    centre_weight: float = 1.0

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Return a legal move selected by tactical rules."""

        legal_moves = state.legal_moves()

        if not legal_moves:
            raise ValueError(
                "TacticalAgent cannot choose a move from a terminal state."
            )

        # 1. If the current player can win immediately, take the win.
        winning_moves = self.immediate_winning_moves(
            state=state,
            player=state.player_to_move,
        )

        if winning_moves:
            return rng.choice(winning_moves)

        # 2. If the opponent can win immediately, block that square.
        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self.immediate_winning_moves(
            state=state,
            player=opponent,
        )

        if blocking_moves:
            return rng.choice(blocking_moves)

        # 3. Otherwise, evaluate all legal moves using a simple heuristic.
        move_scores = {
            move: self.score_move(state, move)
            for move in legal_moves
        }

        best_score = max(move_scores.values())
        best_moves = [
            move
            for move, score in move_scores.items()
            if score == best_score
        ]

        return rng.choice(best_moves)

    @staticmethod
    def immediate_winning_moves(
        state: TicTacToeState,
        player: Mark,
    ) -> list[Move]:
        """
        Return moves that would immediately win for the supplied player.

        The actual state may say it is another player's turn. For threat
        detection, we temporarily replace player_to_move and ask what would
        happen if the inspected player moved next.
        """

        player_state = replace(state, player_to_move=player)
        winning_moves = []

        for move in state.legal_moves():
            next_state = player_state.apply_move(move)

            if next_state.winner() == player:
                winning_moves.append(move)

        return winning_moves

    def score_move(
        self,
        state: TicTacToeState,
        move: Move,
    ) -> float:
        """
        Score a non-immediate move.

        The score combines:
            - how much the move extends the current player's lines;
            - how much it blocks the opponent's lines;
            - a small preference for central squares.
        """

        player = state.player_to_move
        opponent = Mark(-int(player))

        own_score = self._best_line_length_after_move(
            state=state,
            move=move,
            mark=player,
        )

        opponent_score = self._best_line_length_after_move(
            state=state,
            move=move,
            mark=opponent,
        )

        # Squaring makes longer lines much more important than shorter
        # lines. For example, length 3 is worth much more than length 2.
        line_score = (
            self.own_line_weight * (own_score ** 2)
            + self.opponent_line_weight * (opponent_score ** 2)
        )

        return line_score + self.centre_weight * self._centre_bonus(
            state,
            move,
        )

    # Backwards-compatible aliases for older tests or experiments that may
    # still call the private helper names. New code should use the public
    # immediate_winning_moves() and score_move() methods above.
    _immediate_winning_moves = immediate_winning_moves
    _score_move = score_move

    @staticmethod
    def _best_line_length_after_move(
        state: TicTacToeState,
        move: Move,
        mark: Mark,
    ) -> int:
        """
        Return the longest same-mark line created by placing mark at move.

        This does not apply the move to the actual game state. It simply
        counts neighbouring marks around the candidate square.
        """

        directions = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )

        best = 1

        for row_step, col_step in directions:
            line_length = 1
            line_length += TacticalAgent._count_direction(
                state=state,
                start=move,
                row_step=row_step,
                col_step=col_step,
                mark=mark,
            )
            line_length += TacticalAgent._count_direction(
                state=state,
                start=move,
                row_step=-row_step,
                col_step=-col_step,
                mark=mark,
            )

            best = max(best, line_length)

        return best

    @staticmethod
    def _count_direction(
        state: TicTacToeState,
        start: Move,
        row_step: int,
        col_step: int,
        mark: Mark,
    ) -> int:
        """Count consecutive matching marks from a start square."""

        count = 0
        row = start.row + row_step
        col = start.col + col_step

        while 0 <= row < state.rows and 0 <= col < state.cols:
            if state.cell_at(Move(row, col)) != mark:
                break

            count += 1
            row += row_step
            col += col_step

        return count

    @staticmethod
    def _centre_bonus(
        state: TicTacToeState,
        move: Move,
    ) -> float:
        """
        Give a small bonus to moves close to the board centre.

        The value is highest at the centre and decreases with squared
        distance. It is only a tie-breaker compared with tactical line
        scores.
        """

        centre_row = (state.rows - 1) / 2
        centre_col = (state.cols - 1) / 2

        squared_distance = (
            (move.row - centre_row) ** 2
            + (move.col - centre_col) ** 2
        )

        return -squared_distance

    def __str__(self) -> str:
        """Return a short human-readable name for traces."""

        return "TacticalAgent"

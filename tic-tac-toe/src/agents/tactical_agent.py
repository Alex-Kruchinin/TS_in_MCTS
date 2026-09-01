from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from src.games.tic_tac_toe import Mark, Move, TicTacToeState


@dataclass(frozen=True, slots=True)
class TacticalAgent:
    """Rule-based agent for testing.

    It handles immediate wins, blocks and optional forks, then scores the
    remaining moves by lines and centre distance
    """

    # Line building matters more than centre position.
    own_line_weight: float = 10.0
    opponent_line_weight: float = 8.0
    centre_weight: float = 1.0

    # Optional fork handling for a stronger benchmark.
    use_fork_detection: bool = False
    fork_threshold: int = 2

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Return a legal move selected by rules"""

        legal_moves = state.legal_moves()

        if not legal_moves:
            raise ValueError(
                "TacticalAgent cannot choose a move from a terminal state."
            )

        # 1. Take an immediate win
        winning_moves = self.immediate_winning_moves(
            state=state,
            player=state.player_to_move,
        )

        if winning_moves:
            return rng.choice(winning_moves)

        # 2. Block an immediate loss
        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self.immediate_winning_moves(
            state=state,
            player=opponent,
        )

        if blocking_moves:
            return rng.choice(blocking_moves)

        # 3. Create or block a fork when enabled
        if self.use_fork_detection:
            own_forks = self.fork_creating_moves(
                state=state,
                player=state.player_to_move,
            )
            if own_forks:
                return self._best_scored_move(state, own_forks, rng)

            opponent_forks = self.fork_creating_moves(
                state=state,
                player=opponent,
            )
            if opponent_forks:
                return self._best_scored_move(state, opponent_forks, rng)

        # 4. Score the remaining moves
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
        """Return immediate winning moves for ``player``.

        Temporarily make ``player`` the next player when checking threats
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
        """Score a move by line growth, blocking and centre distance"""

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

        # Squaring gives longer lines more importance
        line_score = (
            self.own_line_weight * (own_score ** 2)
            + self.opponent_line_weight * (opponent_score ** 2)
        )

        return line_score + self.centre_weight * self._centre_bonus(
            state,
            move,
        )

    def _best_scored_move(
        self,
        state: TicTacToeState,
        moves: list[Move],
        rng: Random,
    ) -> Move:
        """Choose the highest-scoring move from a restricted candidate set."""

        scores = {move: self.score_move(state, move) for move in moves}
        best_score = max(scores.values())
        best_moves = [
            move
            for move, score in scores.items()
            if score == best_score
        ]
        return rng.choice(best_moves)

    def fork_creating_moves(
        self,
        state: TicTacToeState,
        player: Mark,
    ) -> list[Move]:
        """Return moves that create at least two winning threats."""

        if not self.use_fork_detection:
            return []

        legal_moves = state.legal_moves()
        player_state = (
            state
            if state.player_to_move == player
            else replace(state, player_to_move=player)
        )

        fork_moves: list[Move] = []

        for move in legal_moves:
            # Do not count immediate wins as forks.
            if self._best_line_length_after_move(
                state=state,
                move=move,
                mark=player,
            ) >= state.win_length:
                continue

            next_state = player_state.apply_move(move)
            future_wins = 0

            for future_move in next_state.legal_moves():
                if self._best_line_length_after_move(
                    state=next_state,
                    move=future_move,
                    mark=player,
                ) >= next_state.win_length:
                    future_wins += 1
                    if future_wins >= self.fork_threshold:
                        fork_moves.append(move)
                        break

        return fork_moves

    # Aliases retained for older tests and experiments.
    _immediate_winning_moves = immediate_winning_moves
    _score_move = score_move

    @staticmethod
    def _best_line_length_after_move(
        state: TicTacToeState,
        move: Move,
        mark: Mark,
    ) -> int:
        """Return the longest line formed by placing ``mark`` at ``move`` """

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
        """Count consecutive matching marks from a start square"""

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
        """Return a small bonus for moves near the board centre"""

        centre_row = (state.rows - 1) / 2
        centre_col = (state.cols - 1) / 2

        squared_distance = (
            (move.row - centre_row) ** 2
            + (move.col - centre_col) ** 2
        )

        return -squared_distance

    def __str__(self) -> str:
        """Return a short readable name for traces"""

        if self.use_fork_detection:
            return "TacticalAgent(forks=True)"

        return "TacticalAgent"

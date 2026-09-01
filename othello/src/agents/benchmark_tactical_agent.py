from __future__ import annotations

import random
from dataclasses import dataclass

from game import Action, DIRECTIONS, Disc, Move, OthelloState, PASS


@dataclass(frozen=True, slots=True)
class BenchmarkWeights:
    """Weights for the intermediate rule-based benchmark"""

    corner: float
    opponent_corner_reply: float
    x_square: float
    c_square: float
    edge: float
    opponent_mobility: float
    forced_pass: float
    frontier: float
    stable_edge: float
    opening_flip: float
    middle_flip: float
    endgame_flip: float


PROFILES: dict[str, BenchmarkWeights] = {
    "easy": BenchmarkWeights(
        corner=1000.0,
        opponent_corner_reply=0.0,
        x_square=350.0,
        c_square=160.0,
        edge=60.0,
        opponent_mobility=0.0,
        forced_pass=0.0,
        frontier=0.0,
        stable_edge=0.0,
        opening_flip=-1.0,
        middle_flip=1.0,
        endgame_flip=8.0,
    ),
    "medium": BenchmarkWeights(
        corner=1000.0,
        opponent_corner_reply=450.0,
        x_square=380.0,
        c_square=180.0,
        edge=65.0,
        opponent_mobility=12.0,
        forced_pass=140.0,
        frontier=0.0,
        stable_edge=0.0,
        opening_flip=-1.0,
        middle_flip=1.0,
        endgame_flip=8.0,
    ),
    "hard": BenchmarkWeights(
        corner=1000.0,
        opponent_corner_reply=600.0,
        x_square=450.0,
        c_square=220.0,
        edge=70.0,
        opponent_mobility=18.0,
        forced_pass=200.0,
        frontier=6.0,
        stable_edge=25.0,
        opening_flip=-1.0,
        middle_flip=2.0,
        endgame_flip=10.0,
    ),
}


class BenchmarkTacticalAgent:
    """Intermediate rule-based Othello opponent with no search.

    It uses selected positional features from TacticalAgent. A seeded mistake
    chooses among the top three moves to control difficulty reproducibly.
    """

    def __init__(
        self,
        *,
        profile: str = "medium",
        mistake_probability: float = 0.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}. "
                f"Choose from {tuple(PROFILES)}."
            )

        if not 0.0 <= mistake_probability <= 1.0:
            raise ValueError(
                "mistake_probability must be between 0 and 1."
            )

        self.profile = profile
        self.weights = PROFILES[profile]
        self.mistake_probability = mistake_probability
        self._rng = random.Random(seed)
        self._name = (
            name
            or f"BenchmarkTacticalAgent({profile}, "
               f"mistake={mistake_probability:g})"
        )

        self.last_action_scores: dict[Action, float] = {}

    @property
    def name(self) -> str:
        return self._name

    def choose_action(self, state: OthelloState) -> Action:
        if state.is_terminal():
            raise ValueError(
                "Cannot choose an action from a terminal state."
            )

        legal_actions = state.legal_actions()

        if legal_actions == (PASS,):
            self.last_action_scores = {PASS: 0.0}
            return PASS

        player = state.current_player

        scored = [
            (self._action_score(state, action, player), action)
            for action in legal_actions
        ]

        self.last_action_scores = {
            action: score
            for score, action in scored
        }

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Break exact best-score ties randomly.
        best_score = scored[0][0]
        best_actions = [
            action
            for score, action in scored
            if score == best_score
        ]

        if (
            self.mistake_probability <= 0.0
            or len(scored) == 1
            or self._rng.random() >= self.mistake_probability
        ):
            return self._rng.choice(best_actions)

        # A mistake chooses from the top three actions.
        top_count = min(3, len(scored))
        return self._rng.choice(
            [action for _, action in scored[:top_count]]
        )

    def _action_score(
        self,
        state: OthelloState,
        move: Move,
        player: Disc,
    ) -> float:
        weights = self.weights
        score = 0.0

        corners = set(self._corners(state))
        x_squares, c_squares = self._danger_squares(state)

        if move in corners:
            score += weights.corner

        if move in x_squares:
            score -= weights.x_square

        if move in c_squares:
            score -= weights.c_square

        if self._is_edge(state, move):
            score += weights.edge

        next_state = state.apply(move)
        opponent = player.opponent

        opponent_moves = next_state.legal_placements(opponent)

        # Avoid giving the opponent a corner.
        opponent_corner_replies = sum(
            reply in corners
            for reply in opponent_moves
        )
        score -= (
            weights.opponent_corner_reply
            * opponent_corner_replies
        )

        # Limit the opponent's mobility.
        score -= (
            weights.opponent_mobility
            * len(opponent_moves)
        )

        # Reward a forced pass.
        if (
            not opponent_moves
            and next_state.legal_placements(player)
        ):
            score += weights.forced_pass

        # Add hard-profile positional terms.
        if weights.frontier:
            score += weights.frontier * (
                self._frontier_count(next_state, opponent)
                - self._frontier_count(next_state, player)
            )

        if weights.stable_edge:
            score += weights.stable_edge * (
                self._stable_edge_count(next_state, player)
                - self._stable_edge_count(next_state, opponent)
            )

        flips = len(
            state.captured_indices(
                move,
                player,
            )
        )

        occupied = (
            state.rows * state.cols
            - state.disc_count(Disc.EMPTY)
        )
        occupied_ratio = occupied / (
            state.rows * state.cols
        )

        if occupied_ratio < 0.40:
            flip_weight = weights.opening_flip
        elif occupied_ratio < 0.75:
            flip_weight = weights.middle_flip
        else:
            flip_weight = weights.endgame_flip

        score += flip_weight * flips

        return score

    @staticmethod
    def _corners(
        state: OthelloState,
    ) -> tuple[Move, Move, Move, Move]:
        return (
            Move(0, 0),
            Move(0, state.cols - 1),
            Move(state.rows - 1, 0),
            Move(state.rows - 1, state.cols - 1),
        )

    @staticmethod
    def _is_edge(
        state: OthelloState,
        move: Move,
    ) -> bool:
        return (
            move.row in (0, state.rows - 1)
            or move.col in (0, state.cols - 1)
        )

    @staticmethod
    def _danger_squares(
        state: OthelloState,
    ) -> tuple[set[Move], set[Move]]:
        last_row = state.rows - 1
        last_col = state.cols - 1

        corner_layout = (
            (
                Move(0, 0),
                Move(1, 1),
                (Move(0, 1), Move(1, 0)),
            ),
            (
                Move(0, last_col),
                Move(1, last_col - 1),
                (
                    Move(0, last_col - 1),
                    Move(1, last_col),
                ),
            ),
            (
                Move(last_row, 0),
                Move(last_row - 1, 1),
                (
                    Move(last_row - 1, 0),
                    Move(last_row, 1),
                ),
            ),
            (
                Move(last_row, last_col),
                Move(last_row - 1, last_col - 1),
                (
                    Move(last_row - 1, last_col),
                    Move(last_row, last_col - 1),
                ),
            ),
        )

        x_squares: set[Move] = set()
        c_squares: set[Move] = set()

        for corner, x_square, adjacent_c_squares in corner_layout:
            if (
                state.disc_at(
                    corner.row,
                    corner.col,
                )
                is Disc.EMPTY
            ):
                x_squares.add(x_square)
                c_squares.update(
                    adjacent_c_squares
                )

        return x_squares, c_squares

    @staticmethod
    def _frontier_count(
        state: OthelloState,
        player: Disc,
    ) -> int:
        count = 0

        for index, disc in enumerate(state.board):
            if disc is not player:
                continue

            move = state.index_to_move(index)

            for dr, dc in DIRECTIONS:
                row = move.row + dr
                col = move.col + dc

                if (
                    state.in_bounds(row, col)
                    and state.disc_at(row, col)
                    is Disc.EMPTY
                ):
                    count += 1
                    break

        return count

    def _stable_edge_count(
        self,
        state: OthelloState,
        player: Disc,
    ) -> int:
        """Conservative count of edge discs connected to owned corners."""

        stable: set[Move] = set()

        corner_walks = (
            (
                Move(0, 0),
                ((0, 1), (1, 0)),
            ),
            (
                Move(0, state.cols - 1),
                ((0, -1), (1, 0)),
            ),
            (
                Move(state.rows - 1, 0),
                ((0, 1), (-1, 0)),
            ),
            (
                Move(
                    state.rows - 1,
                    state.cols - 1,
                ),
                ((0, -1), (-1, 0)),
            ),
        )

        for corner, directions in corner_walks:
            if (
                state.disc_at(
                    corner.row,
                    corner.col,
                )
                is not player
            ):
                continue

            stable.add(corner)

            for dr, dc in directions:
                row = corner.row + dr
                col = corner.col + dc

                while (
                    state.in_bounds(row, col)
                    and state.disc_at(row, col)
                    is player
                ):
                    stable.add(
                        Move(row, col)
                    )
                    row += dr
                    col += dc

        return len(stable)

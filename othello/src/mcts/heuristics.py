from __future__ import annotations

import math
import random

from game import Action, Disc, Move, OthelloState, PASS, PassMove


DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class OthelloHeuristicPolicy:
    """Lightweight Othello heuristic for enhanced MCTS.

    It is cheaper than TacticalAgent because expansion and rollout call it repeatedly
    """

    def action_score(self, state: OthelloState, action: Action) -> float:
        if isinstance(action, PassMove):
            return 0.0

        player = state.current_player
        opponent = player.opponent
        next_state = state.apply(action)
        score = 0.0

        corners = set(self.corners(state))
        x_squares, c_squares = self.danger_squares(state)

        if action in corners:
            score += 1_200.0
        elif action in x_squares:
            score -= 650.0
        elif action in c_squares:
            score -= 320.0
        elif self.is_edge(state, action):
            score += 90.0

        # Penalise giving the opponent a corner
        opponent_moves = next_state.legal_placements(opponent)
        opponent_corner_moves = sum(move in corners for move in opponent_moves)
        score -= 700.0 * opponent_corner_moves

        # Prefer low opponent mobility over early disc gains
        score -= 22.0 * len(opponent_moves)

        if not opponent_moves and next_state.legal_placements(player):
            score += 260.0

        score += 12.0 * (
            self.frontier_count(next_state, opponent)
            - self.frontier_count(next_state, player)
        )

        score += 55.0 * (
            self.stable_edge_count(next_state, player)
            - self.stable_edge_count(next_state, opponent)
        )

        occupied = state.rows * state.cols - state.disc_count(Disc.EMPTY)
        phase = occupied / (state.rows * state.cols)
        flips = len(state.captured_indices(action, player))
        if phase < 0.45:
            score -= 2.0 * flips
        elif phase < 0.80:
            score += 2.0 * flips
        else:
            score += 14.0 * flips

        return score

    def fast_action_score(self, state: OthelloState, action: Action) -> float:
        """Cheaper score used repeatedly during rollout."""

        if isinstance(action, PassMove):
            return 0.0

        player = state.current_player
        opponent = player.opponent
        next_state = state.apply(action)
        corners = set(self.corners(state))
        x_squares, c_squares = self.danger_squares(state)

        score = 0.0
        if action in corners:
            score += 1_000.0
        elif action in x_squares:
            score -= 550.0
        elif action in c_squares:
            score -= 260.0
        elif self.is_edge(state, action):
            score += 70.0

        opponent_moves = next_state.legal_placements(opponent)
        score -= 18.0 * len(opponent_moves)
        score -= 600.0 * sum(move in corners for move in opponent_moves)
        if not opponent_moves and next_state.legal_placements(player):
            score += 220.0

        occupied = state.rows * state.cols - state.disc_count(Disc.EMPTY)
        phase = occupied / (state.rows * state.cols)
        flips = len(state.captured_indices(action, player))
        score += (12.0 if phase >= 0.75 else -1.0) * flips
        return score

    def evaluate_state(self, state: OthelloState, player: Disc) -> float:
        """Estimate a non-terminal state and return a reward in [0, 1]"""

        if state.is_terminal():
            return state.result_for(player)

        opponent = player.opponent
        corners = self.corners(state)
        corner_difference = sum(
            state.disc_at(move.row, move.col) is player for move in corners
        ) - sum(
            state.disc_at(move.row, move.col) is opponent for move in corners
        )
        mobility_difference = len(state.legal_placements(player)) - len(
            state.legal_placements(opponent)
        )
        frontier_difference = self.frontier_count(
            state, opponent
        ) - self.frontier_count(state, player)
        stable_difference = self.stable_edge_count(
            state, player
        ) - self.stable_edge_count(state, opponent)
        disc_difference = state.disc_count(player) - state.disc_count(opponent)

        occupied = state.rows * state.cols - state.disc_count(Disc.EMPTY)
        phase = occupied / (state.rows * state.cols)
        disc_weight = -2.0 if phase < 0.45 else (4.0 if phase < 0.8 else 20.0)

        x_squares, c_squares = self.danger_squares(state)
        x_safety = sum(
            state.disc_at(move.row, move.col) is opponent
            for move in x_squares
        ) - sum(
            state.disc_at(move.row, move.col) is player
            for move in x_squares
        )
        c_safety = sum(
            state.disc_at(move.row, move.col) is opponent
            for move in c_squares
        ) - sum(
            state.disc_at(move.row, move.col) is player
            for move in c_squares
        )

        raw_score = (
            450.0 * corner_difference
            + 18.0 * mobility_difference
            + 10.0 * frontier_difference
            + 45.0 * stable_difference
            + 190.0 * x_safety
            + 85.0 * c_safety
            + disc_weight * disc_difference
        )
        return 0.5 + 0.5 * math.tanh(raw_score / 700.0)

    def normalised_prior(self, state: OthelloState, action: Action) -> float:
        """Map an action heuristic to approximately [-1, 1]"""

        return math.tanh(self.action_score(state, action) / 500.0)

    def choose_action(
        self,
        state: OthelloState,
        actions: tuple[Action, ...] | list[Action],
        rng: random.Random,
        *,
        epsilon: float = 0.0,
        fast: bool = False,
    ) -> Action:
        if not actions:
            raise ValueError("Cannot choose from an empty action collection.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between zero and one.")

        if len(actions) == 1:
            return actions[0]
        if rng.random() < epsilon:
            return rng.choice(list(actions))

        scorer = self.fast_action_score if fast else self.action_score
        scored = [(scorer(state, action), action) for action in actions]
        best_score = max(score for score, _ in scored)
        best_actions = [
            action for score, action in scored if score == best_score
        ]
        return rng.choice(best_actions)

    @staticmethod
    def corners(state: OthelloState) -> tuple[Move, Move, Move, Move]:
        return (
            Move(0, 0),
            Move(0, state.cols - 1),
            Move(state.rows - 1, 0),
            Move(state.rows - 1, state.cols - 1),
        )

    @staticmethod
    def is_edge(state: OthelloState, move: Move) -> bool:
        return (
            move.row in (0, state.rows - 1)
            or move.col in (0, state.cols - 1)
        )

    def danger_squares(
        self,
        state: OthelloState,
    ) -> tuple[set[Move], set[Move]]:
        x_squares: set[Move] = set()
        c_squares: set[Move] = set()

        data = (
            (Move(0, 0), Move(1, 1), (Move(0, 1), Move(1, 0))),
            (
                Move(0, state.cols - 1),
                Move(1, state.cols - 2),
                (Move(0, state.cols - 2), Move(1, state.cols - 1)),
            ),
            (
                Move(state.rows - 1, 0),
                Move(state.rows - 2, 1),
                (Move(state.rows - 2, 0), Move(state.rows - 1, 1)),
            ),
            (
                Move(state.rows - 1, state.cols - 1),
                Move(state.rows - 2, state.cols - 2),
                (
                    Move(state.rows - 2, state.cols - 1),
                    Move(state.rows - 1, state.cols - 2),
                ),
            ),
        )

        for corner, x_square, adjacent_c_squares in data:
            if state.disc_at(corner.row, corner.col) is Disc.EMPTY:
                x_squares.add(x_square)
                c_squares.update(adjacent_c_squares)

        return x_squares, c_squares

    @staticmethod
    def frontier_count(state: OthelloState, player: Disc) -> int:
        count = 0
        for row in range(state.rows):
            for col in range(state.cols):
                if state.disc_at(row, col) is not player:
                    continue
                if any(
                    state.in_bounds(row + dr, col + dc)
                    and state.disc_at(row + dr, col + dc) is Disc.EMPTY
                    for dr, dc in DIRECTIONS
                ):
                    count += 1
        return count

    def stable_edge_count(self, state: OthelloState, player: Disc) -> int:
        stable: set[tuple[int, int]] = set()

        corner_walks = (
            ((0, 0), ((0, 1), (1, 0))),
            ((0, state.cols - 1), ((0, -1), (1, 0))),
            ((state.rows - 1, 0), ((0, 1), (-1, 0))),
            (
                (state.rows - 1, state.cols - 1),
                ((0, -1), (-1, 0)),
            ),
        )

        for (corner_row, corner_col), directions in corner_walks:
            if state.disc_at(corner_row, corner_col) is not player:
                continue
            stable.add((corner_row, corner_col))
            for dr, dc in directions:
                row = corner_row + dr
                col = corner_col + dc
                while (
                    state.in_bounds(row, col)
                    and state.disc_at(row, col) is player
                ):
                    stable.add((row, col))
                    row += dr
                    col += dc

        return len(stable)

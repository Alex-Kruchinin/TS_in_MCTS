from __future__ import annotations

import math
import random
from dataclasses import dataclass

from game import Action, DIRECTIONS, Disc, Move, OthelloState, PASS, PassMove


@dataclass(frozen=True, slots=True)
class TacticalWeights:
    """Tunable weights that favour long-term Othello position"""

    corner: float = 1_000.0
    corner_access: float = 450.0
    stable_edge: float = 120.0
    mobility: float = 75.0
    potential_mobility: float = 12.0
    frontier: float = 30.0
    edge: float = 18.0
    empty_corner_x_square: float = 300.0
    empty_corner_c_square: float = 140.0
    forced_pass: float = 250.0


class TacticalAgent:
    """Othello heuristic agent with alpha-beta search"""

    def __init__(
        self,
        search_depth: int = 2,
        *,
        exact_endgame_empty: int = 8,
        weights: TacticalWeights | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if search_depth <= 0:
            raise ValueError("Search depth must be greater than zero.")
        if exact_endgame_empty < 0:
            raise ValueError("Exact endgame threshold cannot be negative.")

        self.search_depth = search_depth
        self.exact_endgame_empty = exact_endgame_empty
        self.weights = weights or TacticalWeights()
        self._rng = random.Random(seed)
        self._name = name or f"TacticalAgent(depth={search_depth})"
        self.last_action_scores: dict[Action, float] = {}

    @property
    def name(self) -> str:
        return self._name

    def choose_action(self, state: OthelloState) -> Action:
        if state.is_terminal():
            raise ValueError("Cannot choose an action from a terminal state.")

        legal_actions = state.legal_actions()
        if legal_actions == (PASS,):
            self.last_action_scores = {PASS: 0.0}
            return PASS

        root_player = state.current_player
        empty_count = state.disc_count(Disc.EMPTY)
        depth = self.search_depth

        # Search the full game near the end
        if empty_count <= self.exact_endgame_empty:
            depth = max(depth, empty_count * 2 + 2)

        scored_actions: list[tuple[float, Action]] = []
        alpha = -math.inf
        beta = math.inf

        for action in self._ordered_actions(state, legal_actions, root_player):
            next_state = state.apply(action)
            score = self._minimax(
                next_state,
                depth=depth - 1,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
            )
            scored_actions.append((score, action))
            alpha = max(alpha, score)

        self.last_action_scores = {
            action: score for score, action in scored_actions
        }
        best_score = max(score for score, _ in scored_actions)
        best_actions = [
            action for score, action in scored_actions if score == best_score
        ]
        return self._rng.choice(best_actions)

    def evaluate(self, state: OthelloState, player: Disc) -> float:
        """Evaluate ``state`` from ``player``'s perspective."""

        if player not in (Disc.BLACK, Disc.WHITE):
            raise ValueError("Evaluation player must be BLACK or WHITE.")

        opponent = player.opponent
        if state.is_terminal():
            disc_difference = (
                state.disc_count(player) - state.disc_count(opponent)
            )
            if disc_difference > 0:
                return 1_000_000.0 + disc_difference * 1_000.0
            if disc_difference < 0:
                return -1_000_000.0 + disc_difference * 1_000.0
            return 0.0

        weights = self.weights
        score = 0.0

        corners = self._corners(state)
        corner_difference = self._owned_count(state, corners, player) - self._owned_count(
            state, corners, opponent
        )
        score += weights.corner * corner_difference

        player_moves = state.legal_placements(player)
        opponent_moves = state.legal_placements(opponent)
        score += weights.mobility * (len(player_moves) - len(opponent_moves))

        player_corner_moves = sum(move in corners for move in player_moves)
        opponent_corner_moves = sum(move in corners for move in opponent_moves)
        score += weights.corner_access * (
            player_corner_moves - opponent_corner_moves
        )

        score += weights.potential_mobility * (
            self._potential_mobility(state, player)
            - self._potential_mobility(state, opponent)
        )

        score += weights.frontier * (
            self._frontier_count(state, opponent)
            - self._frontier_count(state, player)
        )

        score += weights.stable_edge * (
            self._stable_edge_count(state, player)
            - self._stable_edge_count(state, opponent)
        )

        score += weights.edge * (
            self._edge_count(state, player)
            - self._edge_count(state, opponent)
        )

        score += self._empty_corner_danger_score(state, player)

        if not state.legal_placements(state.current_player):
            if state.legal_placements(state.current_player.opponent):
                if state.current_player is opponent:
                    score += weights.forced_pass
                else:
                    score -= weights.forced_pass

        occupied = state.rows * state.cols - state.disc_count(Disc.EMPTY)
        occupied_ratio = occupied / (state.rows * state.cols)
        disc_difference = state.disc_count(player) - state.disc_count(opponent)

        # Disc count matters mainly near the end of the game
        if occupied_ratio < 0.40:
            disc_weight = -3.0
        elif occupied_ratio < 0.75:
            disc_weight = 4.0
        else:
            disc_weight = 24.0
        score += disc_weight * disc_difference

        return score

    def _minimax(
        self,
        state: OthelloState,
        *,
        depth: int,
        root_player: Disc,
        alpha: float,
        beta: float,
    ) -> float:
        if state.is_terminal() or depth <= 0:
            return self.evaluate(state, root_player)

        legal_actions = state.legal_actions()
        maximizing = state.current_player is root_player

        if maximizing:
            value = -math.inf
            for action in self._ordered_actions(
                state, legal_actions, root_player
            ):
                value = max(
                    value,
                    self._minimax(
                        state.apply(action),
                        depth=depth - 1,
                        root_player=root_player,
                        alpha=alpha,
                        beta=beta,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = math.inf
        for action in self._ordered_actions(state, legal_actions, root_player):
            value = min(
                value,
                self._minimax(
                    state.apply(action),
                    depth=depth - 1,
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                ),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _ordered_actions(
        self,
        state: OthelloState,
        actions: tuple[Action, ...],
        root_player: Disc,
    ) -> tuple[Action, ...]:
        """Put strong moves first to improve alpha-beta pruning."""

        if actions == (PASS,):
            return actions

        corners = set(self._corners(state))
        dangerous_x, dangerous_c = self._danger_squares(state)

        def priority(action: Action) -> tuple[int, float]:
            if isinstance(action, PassMove):
                return (-10_000, 0.0)
            if action in corners:
                category = 3
            elif action in dangerous_x:
                category = -2
            elif action in dangerous_c:
                category = -1
            elif self._is_edge(state, action):
                category = 2
            else:
                category = 0

            next_state = state.apply(action)
            heuristic = self.evaluate(next_state, root_player)
            return (category, heuristic)

        reverse = state.current_player is root_player
        return tuple(sorted(actions, key=priority, reverse=reverse))

    @staticmethod
    def _corners(state: OthelloState) -> tuple[Move, Move, Move, Move]:
        return (
            Move(0, 0),
            Move(0, state.cols - 1),
            Move(state.rows - 1, 0),
            Move(state.rows - 1, state.cols - 1),
        )

    @staticmethod
    def _owned_count(
        state: OthelloState,
        positions: tuple[Move, ...],
        player: Disc,
    ) -> int:
        return sum(
            state.disc_at(position.row, position.col) is player
            for position in positions
        )

    @staticmethod
    def _is_edge(state: OthelloState, move: Move) -> bool:
        return (
            move.row in (0, state.rows - 1)
            or move.col in (0, state.cols - 1)
        )

    @staticmethod
    def _edge_count(state: OthelloState, player: Disc) -> int:
        count = 0
        for row in range(state.rows):
            for col in range(state.cols):
                if row not in (0, state.rows - 1) and col not in (
                    0,
                    state.cols - 1,
                ):
                    continue
                if state.disc_at(row, col) is player:
                    count += 1
        return count

    @staticmethod
    def _frontier_count(state: OthelloState, player: Disc) -> int:
        frontier = 0
        for row in range(state.rows):
            for col in range(state.cols):
                if state.disc_at(row, col) is not player:
                    continue
                if any(
                    state.in_bounds(row + dr, col + dc)
                    and state.disc_at(row + dr, col + dc) is Disc.EMPTY
                    for dr, dc in DIRECTIONS
                ):
                    frontier += 1
        return frontier

    @staticmethod
    def _potential_mobility(state: OthelloState, player: Disc) -> int:
        """Empty squares adjacent to at least one opponent disc."""

        opponent = player.opponent
        count = 0
        for row in range(state.rows):
            for col in range(state.cols):
                if state.disc_at(row, col) is not Disc.EMPTY:
                    continue
                if any(
                    state.in_bounds(row + dr, col + dc)
                    and state.disc_at(row + dr, col + dc) is opponent
                    for dr, dc in DIRECTIONS
                ):
                    count += 1
        return count

    def _danger_squares(
        self,
        state: OthelloState,
    ) -> tuple[set[Move], set[Move]]:
        x_squares: set[Move] = set()
        c_squares: set[Move] = set()

        corner_data = (
            (Move(0, 0), Move(1, 1), (Move(0, 1), Move(1, 0))),
            (
                Move(0, state.cols - 1),
                Move(1, state.cols - 2),
                (Move(0, state.cols - 2), Move(1, state.cols - 1)),
            ),
            (
                Move(state.rows - 1, 0),
                Move(state.rows - 2, 1),
                (Move(state.rows - 1, 1), Move(state.rows - 2, 0)),
            ),
            (
                Move(state.rows - 1, state.cols - 1),
                Move(state.rows - 2, state.cols - 2),
                (
                    Move(state.rows - 1, state.cols - 2),
                    Move(state.rows - 2, state.cols - 1),
                ),
            ),
        )

        for corner, x_square, adjacent_c_squares in corner_data:
            if state.disc_at(corner.row, corner.col) is Disc.EMPTY:
                x_squares.add(x_square)
                c_squares.update(adjacent_c_squares)

        return x_squares, c_squares

    def _empty_corner_danger_score(
        self,
        state: OthelloState,
        player: Disc,
    ) -> float:
        opponent = player.opponent
        x_squares, c_squares = self._danger_squares(state)

        score = 0.0
        for square in x_squares:
            occupant = state.disc_at(square.row, square.col)
            if occupant is player:
                score -= self.weights.empty_corner_x_square
            elif occupant is opponent:
                score += self.weights.empty_corner_x_square

        for square in c_squares:
            occupant = state.disc_at(square.row, square.col)
            if occupant is player:
                score -= self.weights.empty_corner_c_square
            elif occupant is opponent:
                score += self.weights.empty_corner_c_square

        return score

    def _stable_edge_count(self, state: OthelloState, player: Disc) -> int:
        """Estimate stability from edge discs connected to owned corners."""

        stable: set[Move] = set()
        corner_walks = (
            (Move(0, 0), ((0, 1), (1, 0))),
            (Move(0, state.cols - 1), ((0, -1), (1, 0))),
            (Move(state.rows - 1, 0), ((0, 1), (-1, 0))),
            (
                Move(state.rows - 1, state.cols - 1),
                ((0, -1), (-1, 0)),
            ),
        )

        for corner, directions in corner_walks:
            if state.disc_at(corner.row, corner.col) is not player:
                continue
            stable.add(corner)
            for dr, dc in directions:
                row = corner.row + dr
                col = corner.col + dc
                while (
                    state.in_bounds(row, col)
                    and state.disc_at(row, col) is player
                ):
                    stable.add(Move(row, col))
                    row += dr
                    col += dc

        return len(stable)

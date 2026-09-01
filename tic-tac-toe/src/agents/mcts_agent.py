from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random
from typing import Literal

from src.agents.mcts_node import MCTSNode
from src.agents.selection_policies import (
    ProgressiveBiasUCTSelectionPolicy,
    ThompsonSamplingSelectionPolicy,
    TreeSelectionPolicy,
    UCTSelectionPolicy,
)
from src.games.tic_tac_toe import Mark, Move, TicTacToeState


RolloutPolicyName = Literal["random", "internal_heuristic"]


@dataclass(slots=True)
class _SearchHeuristicCache:
    """Store reusable heuristic results during one MCTS search.

    When the same board position is checked again, MCTS can reuse a saved
    result instead of calculating it again. The cache is cleared after the
    agent chooses its real move.
    """

    legal_moves: dict[TicTacToeState, tuple[Move, ...]] = field(
        default_factory=dict
    )
    immediate_wins: dict[
        tuple[TicTacToeState, Mark], tuple[Move, ...]
    ] = field(default_factory=dict)
    future_threat_counts: dict[
        tuple[TicTacToeState, Move, Mark], int
    ] = field(default_factory=dict)
    raw_move_scores: dict[
        tuple[TicTacToeState, Move, bool], float
    ] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MCTSAgent:
    """Monte Carlo Tree Search agent with policies

    Each search runs a fixed number of iterations with four stages:
        1. selection
        2. expansion
        3. rollout
        4. backpropagation

    """

    simulations: int = 1_000
    selection_policy: TreeSelectionPolicy = field(
        default_factory=UCTSelectionPolicy
    )
    rollout_policy: RolloutPolicyName = "random"
    use_tactical_guard: bool = True
    use_heuristic_expansion: bool = False

    # Weights used by the internal move heuristic
    own_line_weight: float = 10.0
    opponent_line_weight: float = 8.0
    centre_weight: float = 1.0

    # Thompson Sampling prior. Beta(1, 1) is neutral
    thompson_prior_alpha: float = 1.0
    thompson_prior_beta: float = 1.0

    # A fork creates at least two winning threats for the player's next turn
    use_fork_detection: bool = False
    own_fork_weight: float = 30.0
    opponent_fork_weight: float = 40.0
    fork_threshold: int = 2


    # Raises an error if invalid number or policy is suplied
    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be a positive integer.")

        if self.rollout_policy not in {"random", "internal_heuristic"}:
            raise ValueError(
                "rollout_policy must be 'random' or 'internal_heuristic'."
            )

        if (
            not math.isfinite(self.thompson_prior_alpha)
            or self.thompson_prior_alpha <= 0
        ):
            raise ValueError("thompson_prior_alpha must be positive and finite.")

        if (
            not math.isfinite(self.thompson_prior_beta)
            or self.thompson_prior_beta <= 0
        ):
            raise ValueError("thompson_prior_beta must be positive and finite.")

        if self.fork_threshold < 2:
            raise ValueError("fork_threshold must be at least 2.")

    @classmethod
    def baseline_uct(
        cls,
        simulations: int = 1_000,
        exploration_constant: float = math.sqrt(2.0),
    ) -> MCTSAgent:
        """Create UCT with random expansion, random rollout."""

        return cls(
            simulations=simulations,
            selection_policy=UCTSelectionPolicy(
                exploration_constant=exploration_constant,
            ),
            rollout_policy="random",
            use_tactical_guard=False,
            use_heuristic_expansion=False,
        )

    @classmethod
    def enhanced_uct(
        cls,
        simulations: int = 1_000,
        exploration_constant: float = math.sqrt(2.0),
        heuristic_weight: float = 0.3,
        use_fork_detection: bool = True,
    ) -> MCTSAgent:
        """Create UCT with internal tactical and search heuristics.

        It uses optional progressive-bias selection, heuristic expansion and rollout,
        a tactical guard, and optional fork detection.
        """

        return cls(
            simulations=simulations,
            selection_policy=ProgressiveBiasUCTSelectionPolicy(
                exploration_constant=exploration_constant,
                heuristic_weight=heuristic_weight,
            ),
            rollout_policy="internal_heuristic",
            use_tactical_guard=True,
            use_heuristic_expansion=True,
            use_fork_detection=use_fork_detection,
        )


    @classmethod
    def progressive_uct_random(
        cls,
        simulations: int = 1_000,
        exploration_constant: float = math.sqrt(2.0),
        heuristic_weight: float = 0.3,
    ) -> MCTSAgent:
        """Create progressive-bias UCT with random MCTS.

        Expansion and rollout are random.
        Expanded children still store the heuristic value needed
        by progressive-bias selection.
        """

        return cls(
            simulations=simulations,
            selection_policy=ProgressiveBiasUCTSelectionPolicy(
                exploration_constant=exploration_constant,
                heuristic_weight=heuristic_weight,
            ),
            rollout_policy="random",
            use_tactical_guard=False,
            use_heuristic_expansion=False,
            use_fork_detection=False,
        )

    @classmethod
    def baseline_thompson(
        cls,
        simulations: int = 1_000,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> MCTSAgent:
        """Create Thompson MCTS with random expansion and rollout."""

        return cls(
            simulations=simulations,
            selection_policy=ThompsonSamplingSelectionPolicy(),
            rollout_policy="random",
            use_tactical_guard=False,
            use_heuristic_expansion=False,
            thompson_prior_alpha=prior_alpha,
            thompson_prior_beta=prior_beta,
        )

    @classmethod
    def enhanced_thompson(
        cls,
        simulations: int = 1_000,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        use_fork_detection: bool = True,
    ) -> MCTSAgent:
        """Create Thompson MCTS with internal tactical and search heuristics.

        It uses the same enhancements as enhanced UCT but selects tree nodes
        with Thompson Sampling.
        """

        return cls(
            simulations=simulations,
            selection_policy=ThompsonSamplingSelectionPolicy(),
            rollout_policy="internal_heuristic",
            use_tactical_guard=True,
            use_heuristic_expansion=True,
            use_fork_detection=use_fork_detection,
            thompson_prior_alpha=prior_alpha,
            thompson_prior_beta=prior_beta,
        )

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """Choose a move from the supplied state.

        The optional tactical guard runs first. Otherwise, MCTS chooses the
        most visited root child and uses its mean value to break ties.
        """

        if state.is_terminal():
            raise ValueError(
                "MCTSAgent cannot choose a move from a terminal state."
            )

        # Share cached heuristic results between the guard and search.
        cache = _SearchHeuristicCache()

        if self.use_tactical_guard:
            tactical_move = self._tactical_guard_move(
                state,
                rng,
                cache=cache,
            )
            if tactical_move is not None:
                return tactical_move

        root = self.search(state, rng, _cache=cache)

        if not root.children:
            raise RuntimeError("MCTS search produced no root children.")

        selected_child = self._select_final_child(root, rng)

        if selected_child.move is None:
            raise RuntimeError("Selected MCTS child has no incoming move.")

        return selected_child.move

    def search(
        self,
        state: TicTacToeState,
        rng: Random,
        _cache: _SearchHeuristicCache | None = None,
    ) -> MCTSNode:
        """Build and return an MCTS tree rooted at the supplied state."""

        if state.is_terminal():
            raise ValueError("Cannot search from a terminal state.")

        cache = _cache if _cache is not None else _SearchHeuristicCache()

        root = MCTSNode(
            state=state,
            initial_alpha=self.thompson_prior_alpha,
            initial_beta=self.thompson_prior_beta,
        )

        for _ in range(self.simulations):
            node = root

            # 1. Follow children through fully expanded nodes.
            while (
                not node.state.is_terminal()
                and node.is_fully_expanded()
            ):
                node = self.selection_policy.select_child(node, rng)

            # 2. Add one untried move.
            if not node.state.is_terminal():
                move, heuristic_value = self._choose_expansion_move(
                    node,
                    rng,
                    cache=cache,
                )
                node = node.expand(
                    rng=rng,
                    move=move,
                    heuristic_value=heuristic_value,
                )

            # 3. Play a rollout.
            terminal_state = self._rollout(
                node.state,
                rng,
                cache=cache,
            )

            # 4. Update the path with the result.
            self._backpropagate(node, terminal_state)

        return root

    def _tactical_guard_move(
        self,
        state: TicTacToeState,
        rng: Random,
        cache: _SearchHeuristicCache | None = None,
    ) -> Move | None:
        """Return a tactical win or block, or None if neither exists."""

        winning_moves = self._immediate_winning_moves(
            state=state,
            player=state.player_to_move,
            cache=cache,
        )

        if winning_moves:
            return rng.choice(winning_moves)

        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self._immediate_winning_moves(
            state=state,
            player=opponent,
            cache=cache,
        )

        if blocking_moves:
            return rng.choice(blocking_moves)

        if self.use_fork_detection:
            opponent_fork_moves = self._fork_creating_moves(
                state=state,
                player=opponent,
                cache=cache,
            )
            if opponent_fork_moves:
                return rng.choice(opponent_fork_moves)

            own_fork_moves = self._fork_creating_moves(
                state=state,
                player=state.player_to_move,
                cache=cache,
            )
            if own_fork_moves:
                return rng.choice(own_fork_moves)

        return None

    def _choose_expansion_move(
        self,
        node: MCTSNode,
        rng: Random,
        cache: _SearchHeuristicCache | None = None,
    ) -> tuple[Move, float]:
        """Choose an untried move and return it with its heuristic value.

        Random expansion still records a score for progressive bias. Heuristic
        expansion prefers wins, blocks, forks and then the best scored move.
        """

        if not node.untried_moves:
            raise ValueError("Cannot choose expansion move from full node.")

        if not self.use_heuristic_expansion:
            scores = self._normalised_scores(
                state=node.state,
                moves=node.untried_moves,
                cache=cache,
            )
            move = rng.choice(node.untried_moves)
            return move, scores[move]

        # Check tactical moves before scoring every move.
        winning_moves = [
            move
            for move in self._immediate_winning_moves(
                state=node.state,
                player=node.state.player_to_move,
                cache=cache,
            )
            if move in node.untried_moves
        ]

        if winning_moves:
            move = rng.choice(winning_moves)
            return move, 1.0

        opponent = Mark(-int(node.state.player_to_move))
        blocking_moves = [
            move
            for move in self._immediate_winning_moves(
                state=node.state,
                player=opponent,
                cache=cache,
            )
            if move in node.untried_moves
        ]

        if blocking_moves:
            move = rng.choice(blocking_moves)
            return move, 0.9

        if self.use_fork_detection:
            # Check only moves that can still be expanded.
            opponent_fork_moves = self._fork_creating_moves(
                state=node.state,
                player=opponent,
                moves=node.untried_moves,
                cache=cache,
            )
            if opponent_fork_moves:
                move = rng.choice(opponent_fork_moves)
                return move, 0.92

            own_fork_moves = self._fork_creating_moves(
                state=node.state,
                player=node.state.player_to_move,
                moves=node.untried_moves,
                cache=cache,
            )
            if own_fork_moves:
                move = rng.choice(own_fork_moves)
                return move, 0.95

        # Score all remaining non-tactical moves.
        scores = self._normalised_scores(
            state=node.state,
            moves=node.untried_moves,
            cache=cache,
        )
        best_score = max(scores.values())
        best_moves = [
            move
            for move, score in scores.items()
            if math.isclose(score, best_score)
        ]

        move = rng.choice(best_moves)
        return move, scores[move]

    def _normalised_scores(
        self,
        state: TicTacToeState,
        moves: list[Move],
        cache: _SearchHeuristicCache | None = None,
    ) -> dict[Move, float]:
        """Score moves and normalise their values to the range [0, 1]."""

        raw_scores = {
            move: self._score_move(
                state,
                move,
                include_forks=self.use_fork_detection,
                cache=cache,
            )
            for move in moves
        }

        minimum = min(raw_scores.values())
        maximum = max(raw_scores.values())

        if math.isclose(minimum, maximum):
            return {move: 0.0 for move in moves}

        return {
            move: (score - minimum) / (maximum - minimum)
            for move, score in raw_scores.items()
        }

    def _rollout(
        self,
        state: TicTacToeState,
        rng: Random,
        cache: _SearchHeuristicCache | None = None,
    ) -> TicTacToeState:
        """Play a temporary game from state until it ends."""

        rollout_state = state

        while not rollout_state.is_terminal():
            if self.rollout_policy == "random":
                move = rng.choice(
                    self._legal_moves(rollout_state, cache)
                )
            else:
                move = self._choose_heuristic_rollout_move(
                    rollout_state,
                    rng,
                    cache=cache,
                )

            if not rollout_state.is_legal_move(move):
                raise ValueError(
                    "Rollout policy returned an illegal move."
                )

            rollout_state = rollout_state.apply_move(move)

        return rollout_state

    def _choose_heuristic_rollout_move(
        self,
        state: TicTacToeState,
        rng: Random,
        cache: _SearchHeuristicCache | None = None,
    ) -> Move:
        """Choose a rollout move with internal heuristics.

        Priority:
            1. take an immediate win;
            2. block an immediate loss;
            3. choose a highest-scoring move.

        Rollouts skip expensive fork detection.
        """

        legal_moves = self._legal_moves(state, cache)
        if not legal_moves:
            raise ValueError("Cannot roll out from a terminal state.")

        winning_moves = self._immediate_winning_moves(
            state=state,
            player=state.player_to_move,
            cache=cache,
        )
        if winning_moves:
            return rng.choice(winning_moves)

        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self._immediate_winning_moves(
            state=state,
            player=opponent,
            cache=cache,
        )
        if blocking_moves:
            return rng.choice(blocking_moves)

        # Use the cheaper line and centre heuristic during rollouts.
        move_scores = {
            move: self._score_move(
                state,
                move,
                include_forks=False,
                cache=cache,
            )
            for move in legal_moves
        }
        best_score = max(move_scores.values())
        best_moves = [
            move
            for move, score in move_scores.items()
            if math.isclose(score, best_score)
        ]

        return rng.choice(best_moves)

    @staticmethod
    def _legal_moves(
        state: TicTacToeState,
        cache: _SearchHeuristicCache | None = None,
    ) -> tuple[Move, ...]:
        """Return legal moves, reusing a per-search result when available."""

        if cache is None:
            return state.legal_moves()

        cached = cache.legal_moves.get(state)
        if cached is not None:
            return cached

        moves = state.legal_moves()
        cache.legal_moves[state] = moves
        return moves

    def _immediate_winning_moves(
        self,
        state: TicTacToeState,
        player: Mark,
        cache: _SearchHeuristicCache | None = None,
    ) -> list[Move]:
        """Return moves that immediately win for ``player``.

        Direct line checks avoid creating temporary game states.
        """

        key = (state, player)
        if cache is not None:
            cached = cache.immediate_wins.get(key)
            if cached is not None:
                return list(cached)

        winning_moves = tuple(
            move
            for move in self._legal_moves(state, cache)
            if self._best_line_length_after_move(
                state=state,
                move=move,
                mark=player,
            ) >= state.win_length
        )

        if cache is not None:
            cache.immediate_wins[key] = winning_moves

        return list(winning_moves)

    def _future_winning_threat_count_after_move(
        self,
        state: TicTacToeState,
        move: Move,
        player: Mark,
        cache: _SearchHeuristicCache | None = None,
    ) -> int:
        """Count the winning threats that ``move`` creates for ``player``.

        A non-winning move is a fork when this count reaches the fork
        threshold. Results are cached during the current search.
        """

        key = (state, move, player)
        if cache is not None:
            cached = cache.future_threat_counts.get(key)
            if cached is not None:
                return cached

        if not state.is_legal_move(move):
            result = 0
        elif self._best_line_length_after_move(
            state=state,
            move=move,
            mark=player,
        ) >= state.win_length:
            # An immediate win is not counted as a fork.
            result = 0
        else:
            player_state = (
                state
                if state.player_to_move == player
                else replace(state, player_to_move=player)
            )
            next_state = player_state.apply_move(move)
            result = len(
                self._immediate_winning_moves(
                    state=next_state,
                    player=player,
                    cache=cache,
                )
            )

        if cache is not None:
            cache.future_threat_counts[key] = result

        return result

    def _fork_creating_moves(
        self,
        state: TicTacToeState,
        player: Mark,
        moves: list[Move] | tuple[Move, ...] | None = None,
        cache: _SearchHeuristicCache | None = None,
    ) -> list[Move]:
        """Return moves that create a fork for ``player``.

        If ``moves`` is provided, check only that subset.
        """

        if not self.use_fork_detection:
            return []

        candidates = (
            self._legal_moves(state, cache)
            if moves is None
            else moves
        )

        return [
            move
            for move in candidates
            if self._future_winning_threat_count_after_move(
                state=state,
                move=move,
                player=player,
                cache=cache,
            ) >= self.fork_threshold
        ]

    def _score_move(
        self,
        state: TicTacToeState,
        move: Move,
        include_forks: bool = True,
        cache: _SearchHeuristicCache | None = None,
    ) -> float:
        """Score a move with line, fork and centre heuristics."""

        cache_key = (state, move, include_forks)
        if cache is not None:
            cached = cache.raw_move_scores.get(cache_key)
            if cached is not None:
                return cached

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

        line_score = (
            self.own_line_weight * (own_score ** 2)
            + self.opponent_line_weight * (opponent_score ** 2)
        )

        fork_score = 0.0
        if self.use_fork_detection and include_forks:
            own_future_threats = self._future_winning_threat_count_after_move(
                state=state,
                move=move,
                player=player,
                cache=cache,
            )
            opponent_future_threats = (
                self._future_winning_threat_count_after_move(
                    state=state,
                    move=move,
                    player=opponent,
                    cache=cache,
                )
            )

            fork_score = (
                self.own_fork_weight * (own_future_threats ** 2)
                + self.opponent_fork_weight
                * (opponent_future_threats ** 2)
            )

        result = (
            line_score
            + fork_score
            + self.centre_weight * self._centre_bonus(state, move)
        )

        if cache is not None:
            cache.raw_move_scores[cache_key] = result

        return result

    @staticmethod
    def _best_line_length_after_move(
        state: TicTacToeState,
        move: Move,
        mark: Mark,
    ) -> int:
        """Return the longest line formed by placing ``mark`` at ``move``."""

        directions = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )

        best = 1

        for row_step, col_step in directions:
            line_length = 1
            line_length += MCTSAgent._count_direction(
                state=state,
                start=move,
                row_step=row_step,
                col_step=col_step,
                mark=mark,
            )
            line_length += MCTSAgent._count_direction(
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
        """Return a small bonus for moves near the board centre."""

        centre_row = (state.rows - 1) / 2
        centre_col = (state.cols - 1) / 2

        squared_distance = (
            (move.row - centre_row) ** 2
            + (move.col - centre_col) ** 2
        )

        return -squared_distance

    @staticmethod
    def _backpropagate(
        node: MCTSNode,
        terminal_state: TicTacToeState,
    ) -> None:
        """Update every node from the rollout start back to the root."""

        current: MCTSNode | None = node

        while current is not None:
            current.update(terminal_state)
            current = current.parent

    @staticmethod
    def _select_final_child(
        root: MCTSNode,
        rng: Random,
    ) -> MCTSNode:
        """Select the most visited root child.

        Break ties by mean value and then Thompson posterior mean.
        """

        best_key = max(
            (child.visits, child.mean_value, child.posterior_mean)
            for child in root.children.values()
        )

        best_children = [
            child
            for child in root.children.values()
            if (
                child.visits,
                child.mean_value,
                child.posterior_mean,
            ) == best_key
        ]

        return rng.choice(best_children)

    def __str__(self) -> str:
        """Return a short human-readable name for traces."""

        return f"MCTSAgent(simulations={self.simulations})"

    def rollout_policy_name(self) -> str:
        """Return a readable rollout policy name for logs."""

        if self.rollout_policy == "random":
            return "RandomRollout"
        return "InternalHeuristicRollout"

    def detailed_name(self) -> str:
        """Return a fuller configuration description for experiment logs."""

        policy_name = self.selection_policy.__class__.__name__
        rollout_name = self.rollout_policy_name()

        return (
            f"MCTSAgent(simulations={self.simulations}, "
            f"selection={policy_name}, rollout={rollout_name}, "
            f"guard={self.use_tactical_guard}, "
            f"heuristic_expansion={self.use_heuristic_expansion}, "
            f"fork_detection={self.use_fork_detection}, "
            f"prior=Beta({self.thompson_prior_alpha}, "
            f"{self.thompson_prior_beta}))"
        )

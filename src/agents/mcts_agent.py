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


@dataclass(frozen=True, slots=True)
class MCTSAgent:
    """
    Configurable Monte Carlo Tree Search agent.

    The agent performs a fixed number of MCTS iterations whenever it must
    choose a real move. Each iteration contains:
        1. selection;
        2. expansion;
        3. simulation (rollout);
        4. backpropagation.

    The tree-selection policy is pluggable. This means that UCT and
    Thompson Sampling can be compared while keeping the rest of the MCTS
    machinery unchanged.

    Important implementation note:
        The enhanced MCTS variants use only internal rollout and expansion
        heuristics. They do not call the external rule-based opponent agent
        during MCTS search.
    """

    simulations: int = 1_000
    selection_policy: TreeSelectionPolicy = field(
        default_factory=UCTSelectionPolicy
    )
    rollout_policy: RolloutPolicyName = "random"
    use_tactical_guard: bool = True
    use_heuristic_expansion: bool = False

    # Internal heuristic weights used only by MCTSAgent. These are domain
    # features of scalable Tic-Tac-Toe, not calls to another agent.
    own_line_weight: float = 10.0
    opponent_line_weight: float = 8.0
    centre_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be a positive integer.")

        if self.rollout_policy not in {"random", "internal_heuristic"}:
            raise ValueError(
                "rollout_policy must be 'random' or 'internal_heuristic'."
            )

    @classmethod
    def baseline_uct(
        cls,
        simulations: int = 1_000,
        exploration_constant: float = math.sqrt(2.0),
    ) -> MCTSAgent:
        """
        Return the original plain UCT-MCTS baseline.

        This variant uses:
            - UCT selection;
            - random expansion;
            - random rollout;
            - no tactical guard.
        """

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
    ) -> MCTSAgent:
        """
        Return an enhanced UCT-MCTS agent.

        This variant uses:
            - internal tactical guard;
            - progressive-bias UCT selection;
            - internal heuristic-guided expansion;
            - internal heuristic rollout.

        It uses only internal MCTS heuristics during search.
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
        )

    @classmethod
    def baseline_thompson(
        cls,
        simulations: int = 1_000,
    ) -> MCTSAgent:
        """
        Return a plain Thompson-Sampling MCTS baseline.

        This variant differs from baseline_uct only in tree selection:
            - Thompson Sampling selection;
            - random expansion;
            - random rollout;
            - no tactical guard.
        """

        return cls(
            simulations=simulations,
            selection_policy=ThompsonSamplingSelectionPolicy(),
            rollout_policy="random",
            use_tactical_guard=False,
            use_heuristic_expansion=False,
        )

    @classmethod
    def enhanced_thompson(
        cls,
        simulations: int = 1_000,
    ) -> MCTSAgent:
        """
        Return an enhanced Thompson-Sampling MCTS agent.

        This variant keeps the same shared internal MCTS improvements used by
        enhanced UCT:
            - internal tactical guard;
            - internal heuristic-guided expansion;
            - internal heuristic rollout.

        The main difference is the tree-selection rule:
            - Thompson Sampling instead of UCT.

        It uses only internal MCTS heuristics during search.
        """

        return cls(
            simulations=simulations,
            selection_policy=ThompsonSamplingSelectionPolicy(),
            rollout_policy="internal_heuristic",
            use_tactical_guard=True,
            use_heuristic_expansion=True,
        )

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        """
        Search from the supplied state and return the final selected move.

        If tactical guard is enabled, the agent first checks for immediate
        wins and immediate blocks. Otherwise, the final policy uses the
        robust child rule: choose the root child with the highest visit
        count. Mean value is used as a tie-breaker.
        """

        if state.is_terminal():
            raise ValueError(
                "MCTSAgent cannot choose a move from a terminal state."
            )

        if self.use_tactical_guard:
            tactical_move = self._tactical_guard_move(state, rng)
            if tactical_move is not None:
                return tactical_move

        root = self.search(state, rng)

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
    ) -> MCTSNode:
        """Build and return an MCTS tree rooted at the supplied state."""

        if state.is_terminal():
            raise ValueError("Cannot search from a terminal state.")

        root = MCTSNode(state=state)

        for _ in range(self.simulations):
            node = root

            # 1. SELECTION
            # Follow existing children while the node is non-terminal and
            # has no untried legal moves remaining.
            while (
                not node.state.is_terminal()
                and node.is_fully_expanded()
            ):
                node = self.selection_policy.select_child(node, rng)

            # 2. EXPANSION
            # Add one previously untried move to the tree.
            if not node.state.is_terminal():
                move, heuristic_value = self._choose_expansion_move(
                    node,
                    rng,
                )
                node = node.expand(
                    rng=rng,
                    move=move,
                    heuristic_value=heuristic_value,
                )

            # 3. SIMULATION / PLAYOUT
            terminal_state = self._rollout(node.state, rng)

            # 4. BACKPROPAGATION
            self._backpropagate(node, terminal_state)

        return root

    def _tactical_guard_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move | None:
        """
        Return an immediate win/block if one exists, otherwise None.

        This is a cheap tactical safety layer before the more expensive tree
        search. It is deliberately optional so the original vanilla MCTS
        baseline can still be reproduced.
        """

        winning_moves = self._immediate_winning_moves(
            state=state,
            player=state.player_to_move,
        )

        if winning_moves:
            return rng.choice(winning_moves)

        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self._immediate_winning_moves(
            state=state,
            player=opponent,
        )

        if blocking_moves:
            return rng.choice(blocking_moves)

        return None

    def _choose_expansion_move(
        self,
        node: MCTSNode,
        rng: Random,
    ) -> tuple[Move, float]:
        """
        Choose one untried move for expansion and return its heuristic value.

        With heuristic expansion disabled, the move is random but still gets
        a heuristic value so progressive-bias selection can use it later.

        With heuristic expansion enabled, immediate wins are expanded first,
        then immediate blocks, then the highest-scoring internal heuristic
        move.
        """

        if not node.untried_moves:
            raise ValueError("Cannot choose expansion move from full node.")

        scores = self._normalised_scores(
            state=node.state,
            moves=node.untried_moves,
        )

        if not self.use_heuristic_expansion:
            move = rng.choice(node.untried_moves)
            return move, scores[move]

        winning_moves = [
            move
            for move in self._immediate_winning_moves(
                state=node.state,
                player=node.state.player_to_move,
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
            )
            if move in node.untried_moves
        ]

        if blocking_moves:
            move = rng.choice(blocking_moves)
            return move, 0.9

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
    ) -> dict[Move, float]:
        """
        Score moves with internal MCTS heuristics and normalise to [0, 1].

        Normalisation keeps heuristic values usable by progressive bias
        across different board sizes and win lengths.
        """

        raw_scores = {
            move: self._score_move(state, move)
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
    ) -> TicTacToeState:
        """
        Play rollout moves until a terminal state is reached.

        Rollout states are temporary and never change the real match state.
        """

        rollout_state = state

        while not rollout_state.is_terminal():
            if self.rollout_policy == "random":
                move = rng.choice(rollout_state.legal_moves())
            else:
                move = self._choose_heuristic_rollout_move(
                    rollout_state,
                    rng,
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
    ) -> Move:
        """
        Choose a rollout move using internal domain heuristics.

        Priority:
            1. take an immediate win;
            2. block an opponent immediate win;
            3. otherwise select one of the highest-scoring heuristic moves.

        This method deliberately implements the heuristic idea inside MCTS
        itself. This keeps the MCTS implementation self-contained and avoids
        calling another agent during rollouts.
        """

        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("Cannot roll out from a terminal state.")

        winning_moves = self._immediate_winning_moves(
            state=state,
            player=state.player_to_move,
        )
        if winning_moves:
            return rng.choice(winning_moves)

        opponent = Mark(-int(state.player_to_move))
        blocking_moves = self._immediate_winning_moves(
            state=state,
            player=opponent,
        )
        if blocking_moves:
            return rng.choice(blocking_moves)

        move_scores = {
            move: self._score_move(state, move)
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
    def _immediate_winning_moves(
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
        winning_moves: list[Move] = []

        for move in state.legal_moves():
            next_state = player_state.apply_move(move)

            if next_state.winner() == player:
                winning_moves.append(move)

        return winning_moves

    def _score_move(
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
        """
        Select the real move after search using the robust-child rule.

        Main criterion: highest visit count.
        Tie-breakers: mean value, then Thompson posterior mean.
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
            f"heuristic_expansion={self.use_heuristic_expansion})"
        )

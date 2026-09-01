from __future__ import annotations

import math
import random

from game import Action, Disc, OthelloState
from mcts.enhanced_node import EnhancedMCTSNode
from mcts.heuristics import OthelloHeuristicPolicy


class EnhancedUCTSearch:
    """UCT-MCTS with switchable Othello enhancements for ablation studies."""

    def __init__(
        self,
        simulations: int = 500,
        exploration_constant: float = math.sqrt(2.0),
        *,
        progressive_bias_weight: float = 0.75,
        rollout_epsilon: float = 0.15,
        expansion_epsilon: float = 0.05,
        rollout_depth_limit: int = 16,
        guided_expansion: bool = True,
        heuristic_rollouts: bool = True,
        root_safety: bool = True,
        root_lookahead: bool = True,
        heuristic_cutoff: bool = True,
        seed: int | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("Simulations must be greater than zero.")
        if exploration_constant < 0:
            raise ValueError("Exploration constant cannot be negative.")
        if progressive_bias_weight < 0:
            raise ValueError("Progressive-bias weight cannot be negative.")
        if not 0.0 <= rollout_epsilon <= 1.0:
            raise ValueError("Rollout epsilon must be between zero and one.")
        if not 0.0 <= expansion_epsilon <= 1.0:
            raise ValueError("Expansion epsilon must be between zero and one.")
        if rollout_depth_limit <= 0:
            raise ValueError("Rollout depth limit must be greater than zero.")

        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.progressive_bias_weight = progressive_bias_weight
        self.rollout_epsilon = rollout_epsilon
        self.expansion_epsilon = expansion_epsilon
        self.rollout_depth_limit = rollout_depth_limit
        self.guided_expansion = guided_expansion
        self.heuristic_rollouts = heuristic_rollouts
        self.root_safety = root_safety
        self.root_lookahead = root_lookahead
        self.heuristic_cutoff = heuristic_cutoff
        self._rng = random.Random(seed)
        self._policy = OthelloHeuristicPolicy()

    def build_tree(self, root_state: OthelloState) -> EnhancedMCTSNode:
        if root_state.is_terminal():
            raise ValueError("Cannot search from a terminal state.")

        root = EnhancedMCTSNode(root_state)

        for _ in range(self.simulations):
            node = self._select(root)

            if not node.is_terminal and not node.is_fully_expanded:
                effective_expansion_epsilon = (
                    self.expansion_epsilon if self.guided_expansion else 1.0
                )
                node = node.expand(
                    self._policy,
                    self._rng,
                    expansion_epsilon=effective_expansion_epsilon,
                )

            black_value, white_value = self._rollout_values(node.state)
            node.backpropagate_values(
                black_value=black_value,
                white_value=white_value,
            )

        return root

    def choose_action(
        self,
        root_state: OthelloState,
    ) -> tuple[Action, EnhancedMCTSNode]:
        root = self.build_tree(root_state)
        return self.best_root_action(root), root

    def best_root_action(self, root: EnhancedMCTSNode) -> Action:
        if not root.children:
            raise ValueError("The root has no expanded children.")

        eligible = list(root.children)

        if self.root_safety:
            corners = set(self._policy.corners(root.state))
            corner_children = [
                child for child in eligible if child.action in corners
            ]

            if corner_children:
                eligible = corner_children
            else:
                x_squares, c_squares = self._policy.danger_squares(root.state)
                safe_children = [
                    child
                    for child in eligible
                    if child.action not in x_squares
                    and child.action not in c_squares
                ]
                if safe_children:
                    eligible = safe_children

        maximum_visits = max(child.visits for child in eligible)
        scored_children: list[tuple[float, EnhancedMCTSNode]] = []
        root_player = root.state.current_player

        for child in eligible:
            visit_share = child.visits / maximum_visits
            if self.root_lookahead:
                lookahead_value = self._root_lookahead_value(
                    child.state,
                    root_player,
                )
                combined_score = (
                    0.45 * child.mean_value
                    + 0.40 * lookahead_value
                    + 0.15 * visit_share
                )
            else:
                combined_score = (
                    0.75 * child.mean_value
                    + 0.25 * visit_share
                )
            scored_children.append((combined_score, child))

        best_score = max(score for score, _ in scored_children)
        candidates = [
            child
            for score, child in scored_children
            if math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
        ]
        selected = self._rng.choice(candidates)
        if selected.action is None:
            raise RuntimeError("A root child must have an incoming action.")
        return selected.action

    def _root_lookahead_value(
        self,
        state_after_root_action: OthelloState,
        root_player: Disc,
    ) -> float:
        if state_after_root_action.is_terminal():
            return state_after_root_action.result_for(root_player)

        opponent_actions = state_after_root_action.legal_actions()
        reply_values = [
            self._policy.evaluate_state(
                state_after_root_action.apply(action),
                root_player,
            )
            for action in opponent_actions
        ]
        return min(reply_values)

    def _select(self, root: EnhancedMCTSNode) -> EnhancedMCTSNode:
        node = root
        while (
            not node.is_terminal
            and node.is_fully_expanded
            and node.children
        ):
            node = node.best_child(
                self.exploration_constant,
                self.progressive_bias_weight,
                self._rng,
            )
        return node

    def _rollout_action(
        self,
        state: OthelloState,
        legal_actions: tuple[Action, ...],
    ) -> Action:
        if not self.heuristic_rollouts:
            return self._rng.choice(legal_actions)

        return self._policy.choose_action(
            state,
            legal_actions,
            self._rng,
            epsilon=self.rollout_epsilon,
            fast=True,
        )

    def _terminal_values(self, state: OthelloState) -> tuple[float, float]:
        return (
            state.result_for(Disc.BLACK),
            state.result_for(Disc.WHITE),
        )

    def _rollout_values(self, start_state: OthelloState) -> tuple[float, float]:
        state = start_state

        if self.heuristic_cutoff:
            for _ in range(self.rollout_depth_limit):
                if state.is_terminal():
                    return self._terminal_values(state)
                action = self._rollout_action(state, state.legal_actions())
                state = state.apply(action)

            if state.is_terminal():
                return self._terminal_values(state)

            black_value = self._policy.evaluate_state(state, Disc.BLACK)
            return black_value, 1.0 - black_value

        while not state.is_terminal():
            action = self._rollout_action(state, state.legal_actions())
            state = state.apply(action)
        return self._terminal_values(state)

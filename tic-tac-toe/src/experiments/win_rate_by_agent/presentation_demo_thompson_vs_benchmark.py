
"""Demonstrate enhanced Thompson MCTS against a tactical benchmark.

The fixed seed makes the presentation repeatable but is not experimental
evidence. Use ``--no-pause`` for a non-interactive run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from random import Random
from typing import Optional


# ============================================================================
# DEMO CONFIGURATION
# ============================================================================

ROWS = 7
COLS = 7
WIN_LENGTH = 5

EMPTY = "."
PLAYER_X = "X"   # rule-based benchmark, moves first
PLAYER_O = "O"   # Enhanced Thompson Sampling, moves second

DEFAULT_SIMULATIONS = 500
DEFAULT_SEED = 10

# Tic-Tac-Toe parameter study: mildly pessimistic prior.
ALPHA_PRIOR = 1.0
BETA_PRIOR = 2.0




ROLLOUT_RANDOM_PROBABILITY = 0.0
EXPANSION_RANDOM_PROBABILITY = 0.00


# ============================================================================
# GAME MODEL
# ============================================================================

def other_player(player: str) -> str:
    return PLAYER_O if player == PLAYER_X else PLAYER_X


def cell_to_row_col(action: int) -> tuple[int, int]:
    """Return zero-indexed row and column."""
    return divmod(action, COLS)


def display_action(action: int) -> str:
    """Human-readable one-indexed coordinates."""
    row, col = cell_to_row_col(action)
    return f"({row + 1}, {col + 1})"


# Precompute every winning line.
WINNING_LINES: list[tuple[int, ...]] = []

for row in range(ROWS):
    for col in range(COLS):
        for d_row, d_col in ((1, 0), (0, 1), (1, 1), (1, -1)):
            end_row = row + (WIN_LENGTH - 1) * d_row
            end_col = col + (WIN_LENGTH - 1) * d_col

            if 0 <= end_row < ROWS and 0 <= end_col < COLS:
                WINNING_LINES.append(
                    tuple(
                        (row + step * d_row) * COLS
                        + (col + step * d_col)
                        for step in range(WIN_LENGTH)
                    )
                )

LINES_BY_CELL: list[list[tuple[int, ...]]] = [
    [] for _ in range(ROWS * COLS)
]

for line in WINNING_LINES:
    for cell in line:
        LINES_BY_CELL[cell].append(line)


@dataclass(frozen=True)
class TicTacToeState:
    board: tuple[str, ...]
    to_move: str = PLAYER_X
    last_action: Optional[int] = None

    @staticmethod
    def initial() -> "TicTacToeState":
        return TicTacToeState(
            board=tuple([EMPTY] * (ROWS * COLS)),
            to_move=PLAYER_X,
            last_action=None,
        )

    def legal_actions(self) -> list[int]:
        return [
            index
            for index, value in enumerate(self.board)
            if value == EMPTY
        ]

    def apply(self, action: int) -> "TicTacToeState":
        if self.board[action] != EMPTY:
            raise ValueError(f"Cell {display_action(action)} is occupied.")

        new_board = list(self.board)
        new_board[action] = self.to_move

        return TicTacToeState(
            board=tuple(new_board),
            to_move=other_player(self.to_move),
            last_action=action,
        )

    def winner(self) -> Optional[str]:
        if self.last_action is None:
            return None

        player_just_moved = other_player(self.to_move)

        for line in LINES_BY_CELL[self.last_action]:
            if all(self.board[cell] == player_just_moved for cell in line):
                return player_just_moved

        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or EMPTY not in self.board


def print_board(state: TicTacToeState) -> None:
    print()
    print("      " + "   ".join(str(col + 1) for col in range(COLS)))
    print("    +" + "---+" * COLS)

    for row in range(ROWS):
        values = [
            state.board[row * COLS + col]
            for col in range(COLS)
        ]
        print(f" {row + 1:>2} | " + " | ".join(values) + " |")
        print("    +" + "---+" * COLS)

    print()


# ============================================================================
# TACTICAL / HEURISTIC FEATURES
# ============================================================================

def would_win(
    state: TicTacToeState,
    action: int,
    player: str,
) -> bool:
    """Would 'player' win immediately by placing a mark at action?"""

    if state.board[action] != EMPTY:
        return False

    for line in LINES_BY_CELL[action]:
        if all(
            cell == action or state.board[cell] == player
            for cell in line
        ):
            return True

    return False


def immediate_winning_actions(
    state: TicTacToeState,
    player: str,
    legal_actions: Optional[list[int]] = None,
) -> list[int]:
    actions = (
        legal_actions
        if legal_actions is not None
        else state.legal_actions()
    )

    return [
        action
        for action in actions
        if would_win(state, action, player)
    ]


def heuristic_score(
    state: TicTacToeState,
    action: int,
    player: str,
) -> float:
    """Score line growth, blocking and centre position."""

    opponent = other_player(player)
    score = 0.0

    for line in LINES_BY_CELL[action]:
        # Score the player's line through this square.
        own_values = [
            player if cell == action else state.board[cell]
            for cell in line
        ]

        if opponent not in own_values:
            own_count = own_values.count(player)
            score += 12.0 * (own_count ** 2)

        # Score the same square as an opponent threat.
        opponent_values = [
            opponent if cell == action else state.board[cell]
            for cell in line
        ]

        if player not in opponent_values:
            opponent_count = opponent_values.count(opponent)
            score += 9.0 * (opponent_count ** 2)

    row, col = cell_to_row_col(action)
    centre = (ROWS - 1) / 2.0

    centre_distance = abs(row - centre) + abs(col - centre)
    score += max(0.0, 6.0 - 0.5 * centre_distance)

    return score


# ============================================================================
# RULE-BASED BENCHMARK
# ============================================================================

class TacticalBenchmarkAgent:
    """Fixed rule-based opponent

    Priorities:
      1. centre on the empty board;
      2. immediate win;
      3. immediate block;
      4. score the remaining moves.
    """

    name = "Rule-based Tactical Benchmark"

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> int:
        legal_actions = state.legal_actions()

        # Open in the centre for this presentation.
        if len(legal_actions) == ROWS * COLS:
            return (ROWS // 2) * COLS + (COLS // 2)

        wins = immediate_winning_actions(
            state,
            state.to_move,
            legal_actions,
        )
        if wins:
            return rng.choice(wins)

        blocks = immediate_winning_actions(
            state,
            other_player(state.to_move),
            legal_actions,
        )
        if blocks:
            return max(
                blocks,
                key=lambda action: heuristic_score(
                    state,
                    action,
                    state.to_move,
                ),
            )

        ranked = sorted(
            (
                heuristic_score(state, action, state.to_move),
                action,
            )
            for action in legal_actions
        )
        ranked.reverse()

        best_score = ranked[0][0]

        # Break close ties with the seeded generator.
        candidates = [
            action
            for score, action in ranked[:3]
            if score >= best_score - 8.0
        ]

        return rng.choice(candidates)


# ============================================================================
# MCTS NODE
# ============================================================================

@dataclass
class MCTSNode:
    state: TicTacToeState
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None

    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    untried_actions: list[int] = field(default_factory=list)

    visits: int = 0
    value_sum: float = 0.0

    alpha: float = ALPHA_PRIOR
    beta: float = BETA_PRIOR

    heuristic_value: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_actions and not self.state.is_terminal():
            self.untried_actions = self.state.legal_actions()

    @property
    def empirical_mean(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


# ============================================================================
# ENHANCED THOMPSON-SAMPLING MCTS
# ============================================================================

class EnhancedThompsonMCTSAgent:
    name = "Enhanced Thompson Sampling MCTS"

    def __init__(
        self,
        simulations: int,
        seed: int,
    ) -> None:
        self.simulations = simulations
        self.rng = Random(seed)
        self.last_root: Optional[MCTSNode] = None

    def _guided_action(
        self,
        state: TicTacToeState,
        legal_actions: list[int],
        random_probability: float,
    ) -> int:
        player = state.to_move

        wins = immediate_winning_actions(
            state,
            player,
            legal_actions,
        )
        if wins:
            return self.rng.choice(wins)

        blocks = immediate_winning_actions(
            state,
            other_player(player),
            legal_actions,
        )
        if blocks:
            return max(
                blocks,
                key=lambda action: heuristic_score(
                    state,
                    action,
                    player,
                ),
            )

        # Keep some search variation.
        if self.rng.random() < random_probability:
            return self.rng.choice(legal_actions)

        ranked = sorted(
            (
                heuristic_score(state, action, player),
                action,
            )
            for action in legal_actions
        )
        ranked.reverse()

        best_score = ranked[0][0]

        # Choose among the strongest close-scoring moves.
        candidates = [
            action
            for score, action in ranked[:3]
            if score >= best_score - 10.0
        ]

        return self.rng.choice(candidates)

    def _select_child_by_thompson(
        self,
        node: MCTSNode,
    ) -> MCTSNode:
        """Select the child with the largest Beta posterior sample."""

        return max(
            node.children.values(),
            key=lambda child: self.rng.betavariate(
                child.alpha,
                child.beta,
            ),
        )

    def _rollout(
        self,
        state: TicTacToeState,
    ) -> TicTacToeState:
        rollout_state = state

        while not rollout_state.is_terminal():
            action = self._guided_action(
                rollout_state,
                rollout_state.legal_actions(),
                ROLLOUT_RANDOM_PROBABILITY,
            )
            rollout_state = rollout_state.apply(action)

        return rollout_state

    @staticmethod
    def _reward_for_node(
        node: MCTSNode,
        winner: Optional[str],
    ) -> float:
        """Return the reward for the player who created this node."""

        player_just_moved = other_player(node.state.to_move)

        if winner is None:
            return 0.5

        return 1.0 if winner == player_just_moved else 0.0

    def choose_move(
        self,
        state: TicTacToeState,
    ) -> int:
        legal_actions = state.legal_actions()

        # Apply tactical checks at the root.
        wins = immediate_winning_actions(
            state,
            state.to_move,
            legal_actions,
        )
        if wins:
            return self.rng.choice(wins)

        blocks = immediate_winning_actions(
            state,
            other_player(state.to_move),
            legal_actions,
        )
        if blocks:
            return max(
                blocks,
                key=lambda action: heuristic_score(
                    state,
                    action,
                    state.to_move,
                ),
            )

        root = MCTSNode(state=state)

        for _ in range(self.simulations):
            node = root

            # 1. Selection.
            while (
                not node.state.is_terminal()
                and not node.untried_actions
                and node.children
            ):
                node = self._select_child_by_thompson(node)

            # 2. Expansion.
            if (
                not node.state.is_terminal()
                and node.untried_actions
            ):
                action = self._guided_action(
                    node.state,
                    node.untried_actions,
                    EXPANSION_RANDOM_PROBABILITY,
                )

                node.untried_actions.remove(action)

                child = MCTSNode(
                    state=node.state.apply(action),
                    parent=node,
                    action=action,
                    heuristic_value=heuristic_score(
                        node.state,
                        action,
                        node.state.to_move,
                    ),
                )

                node.children[action] = child
                node = child

            # 3. Rollout.
            terminal_state = self._rollout(node.state)
            winner = terminal_state.winner()

            # 4. Backpropagation.
            while node is not None:
                node.visits += 1

                reward = self._reward_for_node(
                    node,
                    winner,
                )

                node.value_sum += reward

                # Update the Beta posterior.
                node.alpha += reward
                node.beta += 1.0 - reward

                node = node.parent

        self.last_root = root

        # Choose the most visited root child as the real move.
        chosen_child = max(
            root.children.values(),
            key=lambda child: (
                child.visits,
                child.empirical_mean,
            ),
        )

        return chosen_child.action


# ============================================================================
# SEARCH TREE PRESENTATION
# ============================================================================

def sorted_children(node: MCTSNode) -> list[MCTSNode]:
    return sorted(
        node.children.values(),
        key=lambda child: (
            child.visits,
            child.empirical_mean,
        ),
        reverse=True,
    )


def print_tree_summary(
    root: MCTSNode,
    selected_action: int,
    max_root_children: int = 6,
    max_second_level: int = 3,
) -> None:
    """Print a small part of the actual search tree for one Thompson decision."""

    print("=" * 78)
    print("THOMPSON SEARCH TREE")
    print("=" * 78)
    print(
        f"ROOT: {root.state.to_move} to move "
        f"| simulations = {root.visits}"
    )
    print()

    children = sorted_children(root)[:max_root_children]

    for index, child in enumerate(children):
        is_selected = child.action == selected_action
        branch = "└─" if index == len(children) - 1 else "├─"
        marker = "  <<< SELECTED" if is_selected else ""

        print(
            f"{branch} O {display_action(child.action):>7} "
            f"| visits={child.visits:>3} "
            f"| Beta({child.alpha:.1f},{child.beta:.1f}) "
            f"| posterior mean={child.posterior_mean:.3f} "
            f"| H={child.heuristic_value:.1f}"
            f"{marker}"
        )

        if is_selected:
            replies = sorted_children(child)[:max_second_level]

            for reply_index, reply in enumerate(replies):
                reply_branch = (
                    "   └─"
                    if reply_index == len(replies) - 1
                    else "   ├─"
                )

                print(
                    f"{reply_branch} X reply "
                    f"{display_action(reply.action):>7} "
                    f"| visits={reply.visits:>2} "
                    f"| Beta({reply.alpha:.1f},{reply.beta:.1f})"
                )

    selected = root.children[selected_action]

    print()
    print(
        f"{display_action(selected_action)} received "
        f"{selected.visits} visits, the largest root allocation."
    )
    print("=" * 78)
    print()


# ============================================================================
# DEMO RUNNER
# ============================================================================

def run_demo(
    simulations: int,
    seed: int,
    pause_after_thompson_moves: bool,
) -> None:
    benchmark_rng = Random(seed + 10_000)

    benchmark = TacticalBenchmarkAgent()
    thompson = EnhancedThompsonMCTSAgent(
        simulations=simulations,
        seed=seed,
    )

    state = TicTacToeState.initial()

    print("\n" + "=" * 78)
    print("LIVE DEMO: ENHANCED THOMPSON SAMPLING vs RULE-BASED BENCHMARK")
    print("=" * 78)
    print(
        f"Board: {ROWS}x{COLS} | win length: {WIN_LENGTH} "
        f"| Thompson simulations/move: {simulations}"
    )
    print(
        f"Benchmark = X (first) | Thompson = O (second) | seed = {seed}"
    )

    print_board(state)

    # Move 1: benchmark opens in the centre.
    benchmark_move = benchmark.choose_move(
        state,
        benchmark_rng,
    )
    state = state.apply(benchmark_move)

    print(
        f"[Move 1] Benchmark X -> {display_action(benchmark_move)} "
        "(centre opening)"
    )
    print_board(state)

    # Move 2: first Thompson decision and tree display.
    first_thompson_move = thompson.choose_move(state)

    if thompson.last_root is None:
        raise RuntimeError(
            "No tree was produced for the first Thompson decision."
        )

    print(
        f"[Move 2] Enhanced Thompson O -> "
        f"{display_action(first_thompson_move)}"
    )
    print_tree_summary(
        thompson.last_root,
        selected_action=first_thompson_move,
    )

    state = state.apply(first_thompson_move)
    print("Board after Thompson's first move:")
    print_board(state)

    if pause_after_thompson_moves:
        input(
            "Press ENTER for the benchmark reply and Thompson's second search..."
        )

    # Move 3: benchmark reply.
    benchmark_reply = benchmark.choose_move(
        state,
        benchmark_rng,
    )
    state = state.apply(benchmark_reply)

    print(
        f"[Move 3] Benchmark X -> {display_action(benchmark_reply)}"
    )
    print_board(state)

    # Move 4: second Thompson decision and tree display.
    second_thompson_move = thompson.choose_move(state)

    if thompson.last_root is None:
        raise RuntimeError(
            "No tree was produced for the second Thompson decision."
        )

    print(
        f"[Move 4] Enhanced Thompson O -> "
        f"{display_action(second_thompson_move)}"
    )
    print_tree_summary(
        thompson.last_root,
        selected_action=second_thompson_move,
    )

    state = state.apply(second_thompson_move)
    print("Board after Thompson's second move:")
    print_board(state)

    if pause_after_thompson_moves:
        input(
            "Press ENTER to finish the game..."
        )

    # Finish the game automatically.
    move_number = 4

    print("\nFinishing game...")
    print("-" * 78)

    while not state.is_terminal():
        move_number += 1
        current_player = state.to_move

        if current_player == PLAYER_X:
            action = benchmark.choose_move(
                state,
                benchmark_rng,
            )
            label = "Benchmark X"
        else:
            action = thompson.choose_move(state)
            label = "Enhanced Thompson O"

        print(
            f"[Move {move_number}] {label:<20} "
            f"-> {display_action(action)}"
        )

        state = state.apply(action)

    winner = state.winner()

    print("\n" + "=" * 78)
    print("FINAL BOARD")
    print("=" * 78)
    print_board(state)

    if winner == PLAYER_O:
        print("RESULT: Enhanced Thompson Sampling (O) wins.")
    elif winner == PLAYER_X:
        print("RESULT: Rule-based Benchmark (X) wins.")
    else:
        print("RESULT: Draw.")

    print("=" * 78)

    # The default presentation should end in a Thompson win.
    if (
        seed == DEFAULT_SEED
        and simulations == DEFAULT_SIMULATIONS
        and winner != PLAYER_O
    ):
        raise RuntimeError(
            "The validated default demonstration no longer produced "
            "the expected Thompson win."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Presentation demo of Enhanced Thompson Sampling "
            "against a tactical benchmark."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"MCTS simulations per Thompson decision (default: {DEFAULT_SIMULATIONS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Reproducible presentation seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not pause after the first Thompson tree explanation.",
    )

    args = parser.parse_args()

    run_demo(
        simulations=args.simulations,
        seed=args.seed,
        pause_after_thompson_moves=not args.no_pause,
    )


if __name__ == "__main__":
    main()

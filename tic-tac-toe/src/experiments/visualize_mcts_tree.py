from __future__ import annotations

import sys
from pathlib import Path
from random import Random

# Add the project root for direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.games.tic_tac_toe import Move, TicTacToeState
from src.visualization.mcts_tree_visualizer import export_mcts_tree


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
ROWS = 3
COLS = 3
WIN_LENGTH = 3

SIMULATIONS = 200
SEED = 42

# Use one of:
#   MCTSAgent.baseline_uct(simulations=SIMULATIONS)
#   MCTSAgent.enhanced_uct(simulations=SIMULATIONS)
#   MCTSAgent.baseline_thompson(simulations=SIMULATIONS)
#   MCTSAgent.enhanced_thompson(simulations=SIMULATIONS)
AGENT = MCTSAgent.enhanced_thompson(simulations=SIMULATIONS)

# Keep display limits small for readability.
MAX_DEPTH = 2
TOP_K_CHILDREN = 4
SORT_CHILDREN_BY = "visits"  # "visits", "mean_value", or "posterior_mean"
INCLUDE_BOARDS = True
IMAGE_FORMAT = "png"  # "png", "svg", or "pdf"

OUTPUT_STEM = "mcts_tree_snapshot"
OUTPUT_DIR = Path(__file__).resolve().parent / "mcts_tree_outputs"


# Optional starting position. Leave empty for an empty board.
# Example:
# STARTING_MOVES = [Move(1, 1), Move(0, 0), Move(2, 2)]
STARTING_MOVES: list[Move] = []


def build_starting_state() -> TicTacToeState:
    state = TicTacToeState.new(
        rows=ROWS,
        cols=COLS,
        win_length=WIN_LENGTH,
    )

    for move in STARTING_MOVES:
        state = state.apply_move(move)

    return state


def main() -> None:
    state = build_starting_state()
    rng = Random(SEED)

    if state.is_terminal():
        raise ValueError("The configured starting state is already terminal.")

    print("Building MCTS tree...")
    print(f"Board: {ROWS}x{COLS}, win length: {WIN_LENGTH}")
    print(f"Agent: {AGENT.detailed_name()}")
    print(f"Simulations: {SIMULATIONS}")
    print(f"Seed: {SEED}")
    print("\nStarting board:")
    print(state)

    root = AGENT.search(state=state, rng=rng)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_STEM

    dot_path, image_path = export_mcts_tree(
        root=root,
        output_path=output_path,
        max_depth=MAX_DEPTH,
        top_k_children=TOP_K_CHILDREN,
        include_boards=INCLUDE_BOARDS,
        sort_children_by=SORT_CHILDREN_BY,
        image_format=IMAGE_FORMAT,
    )

    print("\nTree export complete.")
    print(f"DOT file:   {dot_path}")

    if image_path is None:
        print("Image file: not created because Graphviz 'dot' was not found.")
        print("You can still open the .dot file in a Graphviz viewer.")
    else:
        print(f"Image file: {image_path}")

    print("\nMost visited root children:")
    children = sorted(
        root.children.values(),
        key=lambda child: (child.visits, child.mean_value),
        reverse=True,
    )[:TOP_K_CHILDREN]

    for child in children:
        print(
            f"  Move ({child.move.row}, {child.move.col}) | "
            f"visits={child.visits:4d} | "
            f"mean={child.mean_value:.3f} | "
            f"alpha={child.alpha:.2f} | "
            f"beta={child.beta:.2f}"
        )


if __name__ == "__main__":
    main()

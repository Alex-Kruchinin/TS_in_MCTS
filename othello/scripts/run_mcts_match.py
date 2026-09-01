from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from agents import MCTSAgent, RandomAgent
from game import Disc, OthelloState
from matches import print_turn, run_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Othello match between UCT-MCTS and RandomAgent."
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--mcts-colour", choices=("black", "white"), default="black")
    parser.add_argument("--mcts-seed", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcts = MCTSAgent(
        simulations=args.simulations,
        exploration_constant=args.exploration,
        seed=args.mcts_seed,
        name="UCT-MCTS",
    )
    random_agent = RandomAgent(seed=args.random_seed, name="Random")

    mcts_colour = Disc.BLACK if args.mcts_colour == "black" else Disc.WHITE
    black_agent = mcts if mcts_colour is Disc.BLACK else random_agent
    white_agent = random_agent if mcts_colour is Disc.BLACK else mcts

    result = run_match(
        black_agent,
        white_agent,
        initial_state=OthelloState.new(rows=args.rows, cols=args.cols),
        trace_callback=print_turn if args.trace else None,
    )

    print("\nFinal board")
    print(result.final_state)
    print(f"\nBlack: {result.black_count}")
    print(f"White: {result.white_count}")
    print(f"Actions: {result.action_count}")
    print(f"Passes: {result.pass_count}")
    print(
        "Winner: "
        + ("DRAW" if result.winner is None else result.winner.name)
    )


if __name__ == "__main__":
    main()

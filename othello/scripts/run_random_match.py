from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from agents import RandomAgent
from game import Disc, OthelloState
from matches import print_turn, run_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one independent Random-vs-Random Othello match."
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--black-seed", type=int, default=1)
    parser.add_argument("--white-seed", type=int, default=2)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print every board and selected action.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_state = OthelloState.new(rows=args.rows, cols=args.cols)

    result = run_match(
        RandomAgent(seed=args.black_seed, name="Black Random"),
        RandomAgent(seed=args.white_seed, name="White Random"),
        initial_state=initial_state,
        trace_callback=print_turn if args.trace else None,
    )

    print("\nFinal board")
    print(result.final_state)
    print(f"Black: {result.black_count}")
    print(f"White: {result.white_count}")
    print(f"Actions: {result.action_count}")
    print(f"Passes: {result.pass_count}")

    if result.winner is None:
        print("Winner: Draw")
    else:
        print(f"Winner: {result.winner.name}")


if __name__ == "__main__":
    main()

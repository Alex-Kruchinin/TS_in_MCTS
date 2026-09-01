from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import EnhancedMCTSAgent, MCTSAgent
from game import Disc, OthelloState
from matches import run_match


@dataclass
class Summary:
    enhanced_wins: int = 0
    classical_wins: int = 0
    draws: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced Enhanced versus classical UCT-MCTS experiment."
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = Summary()

    for index in range(args.games):
        enhanced_colour = Disc.BLACK if index % 2 == 0 else Disc.WHITE
        enhanced = EnhancedMCTSAgent(
            simulations=args.simulations,
            seed=args.seed + index * 2,
            name="Enhanced UCT-MCTS",
        )
        classical = MCTSAgent(
            simulations=args.simulations,
            seed=args.seed + index * 2 + 1,
            name="Classical UCT-MCTS",
        )

        if enhanced_colour is Disc.BLACK:
            black_agent, white_agent = enhanced, classical
        else:
            black_agent, white_agent = classical, enhanced

        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(args.rows, args.cols),
        )

        if result.winner is None:
            outcome = "DRAW"
            summary.draws += 1
        elif result.winner is enhanced_colour:
            outcome = "ENHANCED WIN"
            summary.enhanced_wins += 1
        else:
            outcome = "CLASSICAL WIN"
            summary.classical_wins += 1

        print(
            f"Game {index + 1:>2}: Enhanced={enhanced_colour.name:<5} | "
            f"{outcome}"
        )

    print("\nSummary")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Simulations per move for both agents: {args.simulations}")
    print(f"Enhanced wins: {summary.enhanced_wins}")
    print(f"Classical wins: {summary.classical_wins}")
    print(f"Draws: {summary.draws}")


if __name__ == "__main__":
    main()

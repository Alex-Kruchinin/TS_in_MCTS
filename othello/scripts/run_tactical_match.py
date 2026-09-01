from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import MCTSAgent, TacticalAgent
from game import Disc, OthelloState
from matches import print_turn, run_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Othello TacticalAgent versus UCT-MCTS match."
    )
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tactical-colour", choices=("black", "white"), default="black")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--exact-endgame-empty", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tactical_colour = (
        Disc.BLACK if args.tactical_colour == "black" else Disc.WHITE
    )

    tactical = TacticalAgent(
        search_depth=args.depth,
        exact_endgame_empty=args.exact_endgame_empty,
        seed=args.seed,
        name="Tactical",
    )
    mcts = MCTSAgent(
        simulations=args.simulations,
        exploration_constant=args.exploration,
        seed=args.seed + 1,
        name="UCT-MCTS",
    )

    if tactical_colour is Disc.BLACK:
        black_agent, white_agent = tactical, mcts
    else:
        black_agent, white_agent = mcts, tactical

    result = run_match(
        black_agent,
        white_agent,
        initial_state=OthelloState.new(rows=args.rows, cols=args.cols),
        trace_callback=print_turn if args.trace else None,
    )

    print("\nFinal board")
    print(result.final_state)
    print(f"Black: {result.black_count}")
    print(f"White: {result.white_count}")
    print(f"Actions: {result.action_count}")
    print(f"Passes: {result.pass_count}")
    print(
        "Winner: "
        + ("DRAW" if result.winner is None else result.winner.name)
    )


if __name__ == "__main__":
    main()

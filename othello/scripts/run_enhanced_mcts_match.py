from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import EnhancedMCTSAgent, MCTSAgent, RandomAgent, TacticalAgent
from game import Disc, OthelloState
from matches import print_turn, run_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one enhanced Othello MCTS match."
    )
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--progressive-bias", type=float, default=0.75)
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--expansion-epsilon", type=float, default=0.05)
    parser.add_argument("--rollout-depth", type=int, default=16)
    parser.add_argument("--enhanced-colour", choices=("black", "white"), default="black")
    parser.add_argument(
        "--opponent",
        choices=("random", "classical", "tactical"),
        default="tactical",
    )
    parser.add_argument("--tactical-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enhanced_colour = (
        Disc.BLACK if args.enhanced_colour == "black" else Disc.WHITE
    )

    enhanced = EnhancedMCTSAgent(
        simulations=args.simulations,
        exploration_constant=args.exploration,
        progressive_bias_weight=args.progressive_bias,
        rollout_epsilon=args.rollout_epsilon,
        expansion_epsilon=args.expansion_epsilon,
        rollout_depth_limit=args.rollout_depth,
        seed=args.seed,
        name="Enhanced UCT-MCTS",
    )

    if args.opponent == "random":
        opponent = RandomAgent(seed=args.seed + 1, name="Random")
    elif args.opponent == "classical":
        opponent = MCTSAgent(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            seed=args.seed + 1,
            name="Classical UCT-MCTS",
        )
    else:
        opponent = TacticalAgent(
            search_depth=args.tactical_depth,
            exact_endgame_empty=8,
            seed=args.seed + 1,
            name="Tactical",
        )

    if enhanced_colour is Disc.BLACK:
        black_agent, white_agent = enhanced, opponent
    else:
        black_agent, white_agent = opponent, enhanced

    print(f"BLACK: {black_agent.name}")
    print(f"WHITE: {white_agent.name}")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Enhanced simulations: {args.simulations}\n")

    result = run_match(
        black_agent,
        white_agent,
        initial_state=OthelloState.new(args.rows, args.cols),
        trace_callback=print_turn if args.trace else None,
    )

    print("Final board")
    print(result.final_state)
    print(f"\nBlack discs: {result.black_count}")
    print(f"White discs: {result.white_count}")
    print(f"Actions: {result.action_count}")
    print(f"Passes: {result.pass_count}")
    print(
        "Winner: "
        + ("DRAW" if result.winner is None else result.winner.name)
    )


if __name__ == "__main__":
    main()

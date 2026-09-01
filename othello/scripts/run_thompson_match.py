from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import (
    EnhancedMCTSAgent,
    EnhancedThompsonMCTSAgent,
    MCTSAgent,
    RandomAgent,
    TacticalAgent,
    ThompsonMCTSAgent,
)
from game import Disc, OthelloState
from matches import print_turn, run_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Othello Thompson-Sampling MCTS match."
    )
    parser.add_argument("--variant", choices=("classical", "enhanced"), default="classical")
    parser.add_argument(
        "--opponent",
        choices=("random", "tactical", "uct", "enhanced-uct"),
        default="uct",
    )
    parser.add_argument("--thompson-colour", choices=("black", "white"), default="black")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta-prior", type=float, default=1.0)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--progressive-bias", type=float, default=0.75)
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--expansion-epsilon", type=float, default=0.05)
    parser.add_argument("--rollout-depth", type=int, default=16)
    parser.add_argument("--tactical-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--show-root", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thompson_colour = (
        Disc.BLACK if args.thompson_colour == "black" else Disc.WHITE
    )

    if args.variant == "classical":
        thompson = ThompsonMCTSAgent(
            simulations=args.simulations,
            alpha_prior=args.alpha_prior,
            beta_prior=args.beta_prior,
            seed=args.seed,
            name="Classical Thompson-MCTS",
        )
    else:
        thompson = EnhancedThompsonMCTSAgent(
            simulations=args.simulations,
            alpha_prior=args.alpha_prior,
            beta_prior=args.beta_prior,
            rollout_epsilon=args.rollout_epsilon,
            expansion_epsilon=args.expansion_epsilon,
            rollout_depth_limit=args.rollout_depth,
            seed=args.seed,
            name="Enhanced Thompson-MCTS",
        )

    if args.opponent == "random":
        opponent = RandomAgent(seed=args.seed + 1, name="Random")
    elif args.opponent == "tactical":
        opponent = TacticalAgent(
            search_depth=args.tactical_depth,
            exact_endgame_empty=8,
            seed=args.seed + 1,
            name="Tactical",
        )
    elif args.opponent == "enhanced-uct":
        opponent = EnhancedMCTSAgent(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            progressive_bias_weight=args.progressive_bias,
            rollout_epsilon=args.rollout_epsilon,
            expansion_epsilon=args.expansion_epsilon,
            rollout_depth_limit=args.rollout_depth,
            seed=args.seed + 1,
            name="Enhanced UCT-MCTS",
        )
    else:
        opponent = MCTSAgent(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            seed=args.seed + 1,
            name="Classical UCT-MCTS",
        )

    if thompson_colour is Disc.BLACK:
        black_agent, white_agent = thompson, opponent
    else:
        black_agent, white_agent = opponent, thompson

    print(f"BLACK: {black_agent.name}")
    print(f"WHITE: {white_agent.name}")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Simulations per MCTS move: {args.simulations}")
    print(f"Thompson prior: Beta({args.alpha_prior}, {args.beta_prior})\n")

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
    print("Winner: " + ("DRAW" if result.winner is None else result.winner.name))

    if args.show_root and thompson.last_root is not None:
        print("\nThompson statistics from its final decision:")
        ordered = sorted(
            thompson.last_root.children,
            key=lambda child: (child.visits, child.mean_value),
            reverse=True,
        )
        for child in ordered:
            print(
                f"{child.action!s:<18} visits={child.visits:<4} "
                f"mean={child.mean_value:.3f} "
                f"alpha={child.alpha:.3f} beta={child.beta:.3f} "
                f"posterior_mean={child.posterior_mean:.3f}"
            )


if __name__ == "__main__":
    main()

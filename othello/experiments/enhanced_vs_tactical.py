from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import EnhancedMCTSAgent, TacticalAgent
from game import Disc, OthelloState
from matches import run_match


@dataclass
class Summary:
    enhanced_wins: int = 0
    tactical_wins: int = 0
    draws: int = 0
    enhanced_discs: int = 0
    tactical_discs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced Enhanced UCT-MCTS versus TacticalAgent experiment."
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--progressive-bias", type=float, default=0.75)
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--expansion-epsilon", type=float, default=0.05)
    parser.add_argument("--rollout-depth", type=int, default=16)
    parser.add_argument("--tactical-depth", type=int, default=2)
    parser.add_argument("--exact-endgame-empty", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument(
        "--fixed-colour",
        choices=("black", "white"),
        default=None,
        help="Keep Enhanced MCTS on one colour instead of alternating.",
    )
    return parser.parse_args()


def enhanced_colour_for_game(index: int, fixed: str | None) -> Disc:
    if fixed == "black":
        return Disc.BLACK
    if fixed == "white":
        return Disc.WHITE
    return Disc.BLACK if index % 2 == 0 else Disc.WHITE


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("Games must be greater than zero.")

    summary = Summary()

    for game_index in range(args.games):
        enhanced_colour = enhanced_colour_for_game(
            game_index,
            args.fixed_colour,
        )
        enhanced = EnhancedMCTSAgent(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            progressive_bias_weight=args.progressive_bias,
            rollout_epsilon=args.rollout_epsilon,
            expansion_epsilon=args.expansion_epsilon,
            rollout_depth_limit=args.rollout_depth,
            seed=args.seed + game_index * 2,
            name="Enhanced UCT-MCTS",
        )
        tactical = TacticalAgent(
            search_depth=args.tactical_depth,
            exact_endgame_empty=args.exact_endgame_empty,
            seed=args.seed + game_index * 2 + 1,
            name="Tactical",
        )

        if enhanced_colour is Disc.BLACK:
            black_agent, white_agent = enhanced, tactical
        else:
            black_agent, white_agent = tactical, enhanced

        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(args.rows, args.cols),
        )

        if enhanced_colour is Disc.BLACK:
            enhanced_discs = result.black_count
            tactical_discs = result.white_count
        else:
            enhanced_discs = result.white_count
            tactical_discs = result.black_count

        summary.enhanced_discs += enhanced_discs
        summary.tactical_discs += tactical_discs

        if result.winner is None:
            outcome = "DRAW"
            summary.draws += 1
        elif result.winner is enhanced_colour:
            outcome = "ENHANCED WIN"
            summary.enhanced_wins += 1
        else:
            outcome = "TACTICAL WIN"
            summary.tactical_wins += 1

        print(
            f"Game {game_index + 1:>2}: "
            f"Enhanced={enhanced_colour.name:<5} | "
            f"Enhanced discs={enhanced_discs:>2} | "
            f"Tactical discs={tactical_discs:>2} | "
            f"{outcome}"
        )

    print("\nSummary")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Enhanced simulations per move: {args.simulations}")
    print(f"Tactical depth: {args.tactical_depth}")
    print(f"Enhanced wins: {summary.enhanced_wins}")
    print(f"Tactical wins: {summary.tactical_wins}")
    print(f"Draws: {summary.draws}")
    print(
        "Average discs: "
        f"Enhanced={summary.enhanced_discs / args.games:.2f}, "
        f"Tactical={summary.tactical_discs / args.games:.2f}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import (
    EnhancedMCTSAgent,
    EnhancedThompsonMCTSAgent,
    MCTSAgent,
    ThompsonMCTSAgent,
)
from game import Disc, OthelloState
from matches import run_match


@dataclass
class Summary:
    thompson_wins: int = 0
    uct_wins: int = 0
    draws: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced Othello Thompson-MCTS versus UCT experiment."
    )
    parser.add_argument("--variant", choices=("classical", "enhanced"), default="classical")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta-prior", type=float, default=2.0)
    parser.add_argument("--exploration", type=float, default=0.5)
    parser.add_argument(
        "--uct-progressive-bias",
        type=float,
        default=0.0,
        help=(
            "Enhanced comparison only. Default 0 isolates Thompson versus UCT "
            "selection while keeping the remaining heuristics equal."
        ),
    )
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--expansion-epsilon", type=float, default=0.05)
    parser.add_argument("--rollout-depth", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional per-game CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.games % 2 != 0:
        raise ValueError("Games must be a positive even number for colour balance.")

    output = args.output or (
        PROJECT_ROOT / "results" / f"{args.variant}_thompson_vs_uct.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = Summary()
    rows: list[dict[str, object]] = []

    for index in range(args.games):
        thompson_colour = Disc.BLACK if index % 2 == 0 else Disc.WHITE
        thompson_seed = args.seed + index * 2
        uct_seed = args.seed + index * 2 + 1

        if args.variant == "classical":
            thompson = ThompsonMCTSAgent(
                simulations=args.simulations,
                alpha_prior=args.alpha_prior,
                beta_prior=args.beta_prior,
                seed=thompson_seed,
                name="Classical Thompson-MCTS",
            )
            uct = MCTSAgent(
                simulations=args.simulations,
                exploration_constant=args.exploration,
                seed=uct_seed,
                name="Classical UCT-MCTS",
            )
        else:
            thompson = EnhancedThompsonMCTSAgent(
                simulations=args.simulations,
                alpha_prior=args.alpha_prior,
                beta_prior=args.beta_prior,
                rollout_epsilon=args.rollout_epsilon,
                expansion_epsilon=args.expansion_epsilon,
                rollout_depth_limit=args.rollout_depth,
                seed=thompson_seed,
                name="Enhanced Thompson-MCTS",
            )
            uct = EnhancedMCTSAgent(
                simulations=args.simulations,
                exploration_constant=args.exploration,
                progressive_bias_weight=args.uct_progressive_bias,
                rollout_epsilon=args.rollout_epsilon,
                expansion_epsilon=args.expansion_epsilon,
                rollout_depth_limit=args.rollout_depth,
                seed=uct_seed,
                name="Enhanced UCT-MCTS",
            )

        if thompson_colour is Disc.BLACK:
            black_agent, white_agent = thompson, uct
        else:
            black_agent, white_agent = uct, thompson

        started = time.perf_counter()
        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(args.rows, args.cols),
        )
        runtime = time.perf_counter() - started

        if result.winner is None:
            outcome = "draw"
            summary.draws += 1
        elif result.winner is thompson_colour:
            outcome = "thompson_win"
            summary.thompson_wins += 1
        else:
            outcome = "uct_win"
            summary.uct_wins += 1

        thompson_discs = (
            result.black_count
            if thompson_colour is Disc.BLACK
            else result.white_count
        )
        uct_discs = (
            result.white_count
            if thompson_colour is Disc.BLACK
            else result.black_count
        )
        rows.append(
            {
                "game": index + 1,
                "variant": args.variant,
                "thompson_colour": thompson_colour.name,
                "outcome": outcome,
                "thompson_discs": thompson_discs,
                "uct_discs": uct_discs,
                "disc_difference": thompson_discs - uct_discs,
                "actions": result.action_count,
                "passes": result.pass_count,
                "runtime_seconds": round(runtime, 6),
                "thompson_seed": thompson_seed,
                "uct_seed": uct_seed,
            }
        )
        print(
            f"Game {index + 1:>2}: Thompson={thompson_colour.name:<5} | "
            f"{outcome.replace('_', ' ').upper():<14} | "
            f"discs {thompson_discs}-{uct_discs}"
        )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    print(f"Variant: {args.variant}")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Simulations per move: {args.simulations}")
    print(f"Thompson wins: {summary.thompson_wins}")
    print(f"UCT wins: {summary.uct_wins}")
    print(f"Draws: {summary.draws}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()

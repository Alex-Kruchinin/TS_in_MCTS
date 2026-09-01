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

from agents import EnhancedThompsonMCTSAgent, TacticalAgent, ThompsonMCTSAgent
from game import Disc, OthelloState
from matches import run_match


@dataclass
class Summary:
    thompson_wins: int = 0
    tactical_wins: int = 0
    draws: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced Othello Thompson-MCTS versus Tactical experiment."
    )
    parser.add_argument("--variant", choices=("classical", "enhanced"), default="classical")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta-prior", type=float, default=1.0)
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--expansion-epsilon", type=float, default=0.05)
    parser.add_argument("--rollout-depth", type=int, default=16)
    parser.add_argument("--tactical-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=14000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.games % 2 != 0:
        raise ValueError("Games must be a positive even number for colour balance.")

    output = args.output or (
        PROJECT_ROOT / "results" / f"{args.variant}_thompson_vs_tactical.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = Summary()
    rows: list[dict[str, object]] = []

    for index in range(args.games):
        thompson_colour = Disc.BLACK if index % 2 == 0 else Disc.WHITE
        thompson_seed = args.seed + index * 2
        tactical_seed = args.seed + index * 2 + 1

        if args.variant == "classical":
            thompson = ThompsonMCTSAgent(
                simulations=args.simulations,
                alpha_prior=args.alpha_prior,
                beta_prior=args.beta_prior,
                seed=thompson_seed,
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
                seed=thompson_seed,
                name="Enhanced Thompson-MCTS",
            )

        tactical = TacticalAgent(
            search_depth=args.tactical_depth,
            exact_endgame_empty=8,
            seed=tactical_seed,
            name="Tactical",
        )

        if thompson_colour is Disc.BLACK:
            black_agent, white_agent = thompson, tactical
        else:
            black_agent, white_agent = tactical, thompson

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
            outcome = "tactical_win"
            summary.tactical_wins += 1

        thompson_discs = (
            result.black_count
            if thompson_colour is Disc.BLACK
            else result.white_count
        )
        tactical_discs = (
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
                "tactical_discs": tactical_discs,
                "disc_difference": thompson_discs - tactical_discs,
                "actions": result.action_count,
                "passes": result.pass_count,
                "runtime_seconds": round(runtime, 6),
                "thompson_seed": thompson_seed,
                "tactical_seed": tactical_seed,
            }
        )
        print(
            f"Game {index + 1:>2}: Thompson={thompson_colour.name:<5} | "
            f"{outcome.replace('_', ' ').upper():<14} | "
            f"discs {thompson_discs}-{tactical_discs}"
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
    print(f"Tactical wins: {summary.tactical_wins}")
    print(f"Draws: {summary.draws}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()

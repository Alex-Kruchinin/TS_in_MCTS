from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import (
    AgentConfiguration,
    ablation_configurations,
    one_factor_configurations,
    run_parameter_study,
)


def _int_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error


def _float_list(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run balanced Enhanced MCTS ablation or one-factor parameter studies "
            "and save per-game and summary CSV files."
        )
    )
    parser.add_argument("--mode", choices=("ablation", "tuning"), default="ablation")
    parser.add_argument(
        "--opponent",
        choices=("classical", "tactical", "random"),
        default="tactical",
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--progressive-bias", type=float, default=0.75)
    parser.add_argument("--rollout-epsilon", type=float, default=0.15)
    parser.add_argument("--expansion-epsilon", type=float, default=0.05)
    parser.add_argument("--rollout-depth", type=int, default=16)
    parser.add_argument(
        "--simulation-values",
        type=_int_list,
        default=(50, 100, 200),
    )
    parser.add_argument(
        "--bias-values",
        type=_float_list,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--rollout-epsilon-values",
        type=_float_list,
        default=(0.0, 0.1, 0.2, 0.3),
    )
    parser.add_argument(
        "--rollout-depth-values",
        type=_int_list,
        default=(8, 12, 16, 24),
    )
    parser.add_argument("--opponent-simulations", type=int, default=None)
    parser.add_argument("--tactical-depth", type=int, default=2)
    parser.add_argument("--exact-endgame-empty", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "enhanced_parameter_study",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = AgentConfiguration(
        name="full_enhanced",
        simulations=args.simulations,
        exploration_constant=args.exploration,
        progressive_bias_weight=args.progressive_bias,
        rollout_epsilon=args.rollout_epsilon,
        expansion_epsilon=args.expansion_epsilon,
        rollout_depth_limit=args.rollout_depth,
    )

    if args.mode == "ablation":
        configurations = ablation_configurations(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            progressive_bias_weight=args.progressive_bias,
            rollout_epsilon=args.rollout_epsilon,
            expansion_epsilon=args.expansion_epsilon,
            rollout_depth_limit=args.rollout_depth,
        )
    else:
        configurations = one_factor_configurations(
            base=base,
            simulation_values=args.simulation_values,
            progressive_bias_values=args.bias_values,
            rollout_epsilon_values=args.rollout_epsilon_values,
            rollout_depth_values=args.rollout_depth_values,
        )

    outputs = run_parameter_study(
        configurations,
        opponent=args.opponent,
        games_per_configuration=args.games,
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
        output_directory=args.output_dir,
        opponent_simulations=args.opponent_simulations,
        opponent_exploration_constant=args.exploration,
        tactical_depth=args.tactical_depth,
        exact_endgame_empty=args.exact_endgame_empty,
        verbose=args.verbose,
    )

    print("\nSaved study outputs")
    print(f"Per-game CSV: {outputs.game_results_csv}")
    print(f"Summary CSV:  {outputs.summary_csv}")
    print(f"Summary table: {outputs.summary_markdown}")
    print(f"Metadata:      {outputs.metadata_json}")


if __name__ == "__main__":
    main()

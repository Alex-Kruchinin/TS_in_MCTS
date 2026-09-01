# ============================================================================
# WAS NOT USED IN FINAL TESTING - Runs resulted in floor effect for classical UCT - all results were at 0%
# ============================================================================

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math

from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    run_win_rate_experiment,
)
from src.agents.mcts_agent import MCTSAgent
from src.agents.tactical_agent import TacticalAgent

# Prevent pytest from treating this long-running experiment as a test file.
__test__ = False

# Test UCT C against a fixed tactical benchmark.
# Expansion and rollout are random, with no guard or fork detection.

BOARD = BoardConfig(rows=7, cols=7, win_length=5)
SIMULATIONS = 400
SEEDS = range(0, 100)
SIDE_MODE = "alternate"

# Include the standard sqrt(2) value alongside game-specific alternatives.
UCT_C_VALUES = [
    0.25,
    0.50,
    0.75,
    1.00,
    math.sqrt(2.0),
    2.00,
    3.00,
]

RESULTS_FOLDER = Path(__file__).resolve().parent / "results" / "parameter_sweeps"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def safe_c_label(c_value: float) -> str:
    if math.isclose(c_value, math.sqrt(2.0)):
        return "sqrt2"
    return str(c_value).replace(".", "_")


def main() -> None:
    print("=" * 80)
    print("UCT C sweep vs TacticalAgent")
    print(f"Board: {BOARD.rows}x{BOARD.cols}, k={BOARD.win_length}")
    print(f"Simulations per MCTS move: {SIMULATIONS}")
    print(f"Seeds: {SEEDS.start}..{SEEDS.stop - 1}")
    print("Each C value is saved to a separate CSV file.")
    print("=" * 80)

    for c_value in UCT_C_VALUES:
        label = safe_c_label(c_value)
        experiment_name = f"uct_c_{label}_vs_tactical"
        output_csv = RESULTS_FOLDER / f"{experiment_name}.csv"

        def make_tested_agent(c: float = c_value):
            return MCTSAgent.baseline_uct(
                simulations=SIMULATIONS,
                exploration_constant=c,
            )

        def make_opponent():
            return TacticalAgent()

        run_win_rate_experiment(
            experiment_name=experiment_name,
            comparison_file=__file__,
            board=BOARD,
            tested_agent_factory=make_tested_agent,
            opponent_factory=make_opponent,
            seeds=SEEDS,
            output_csv=output_csv,
            side_mode=SIDE_MODE,
        )


if __name__ == "__main__":
    main()

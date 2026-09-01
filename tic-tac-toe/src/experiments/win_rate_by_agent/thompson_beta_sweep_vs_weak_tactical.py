from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    run_win_rate_experiment,
)
from src.agents.mcts_agent import MCTSAgent
from src.agents.weak_tactical_agent import WeakTacticalAgent

# Prevents pytest from treating this experiment as a test file
__test__ = False

# Run Thompson prior against the weak tactical benchmark.

BOARD = BoardConfig(rows=7, cols=7, win_length=4)
SIMULATIONS = 400
SEEDS = range(0, 200)
SIDE_MODE = "alternate"

THOMPSON_PRIORS = [
    ("beta_0_5_0_5", 0.5, 0.5),
    ("beta_1_1", 1.0, 1.0),
    ("beta_2_2", 2.0, 2.0),
    ("beta_5_5", 5.0, 5.0),
    ("beta_2_1_optimistic", 2.0, 1.0),
    ("beta_1_2_pessimistic", 1.0, 2.0),
]

RESULTS_FOLDER = Path(__file__).resolve().parent / "results" / "parameter_sweeps"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 80)
    print("Thompson Beta-prior sweep vs WeakTacticalAgent")
    print(f"Board: {BOARD.rows}x{BOARD.cols}, k={BOARD.win_length}")
    print(f"Simulations per MCTS move: {SIMULATIONS}")
    print(f"Seeds: {SEEDS.start}..{SEEDS.stop - 1}")
    print("Each Beta prior is saved to a separate CSV file.")
    print("=" * 80)

    for label, alpha, beta in THOMPSON_PRIORS:
        experiment_name = f"thompson_{label}_vs_weak_tactical"
        output_csv = RESULTS_FOLDER / f"{experiment_name}.csv"

        def make_tested_agent(a: float = alpha, b: float = beta):
            return MCTSAgent.baseline_thompson(
                simulations=SIMULATIONS,
                prior_alpha=a,
                prior_beta=b,
            )

        def make_opponent():
            return WeakTacticalAgent()

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

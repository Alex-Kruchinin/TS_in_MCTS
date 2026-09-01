from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    output_folder,
    run_win_rate_experiment,
)
from src.agents.mcts_agent import MCTSAgent

# Prevents pytest from treating this experiment as a test file
__test__ = False

# Baseline Thompson versus UCT with only the selection policy changed
# Both use random expansion and rollout
BOARD = BoardConfig(rows=7, cols=7, win_length=4)
SIMULATIONS = 100
SEEDS = range(0, 100)
SIDE_MODE = "alternate"

# Selection parameters

UCT_C = 0.5
THOMPSON_PRIOR_ALPHA = 1.0
THOMPSON_PRIOR_BETA = 2.0

EXPERIMENT_NAME = "baseline_thompson_vs_baseline_uct"
OUTPUT_CSV = (
    output_folder()
    / f"{EXPERIMENT_NAME}_{SIMULATIONS}_simulations.csv"
)


def make_tested_agent():
    """Agent whose win rate is measured in the output CSV."""

    return MCTSAgent.baseline_thompson(
        simulations=SIMULATIONS,
        prior_alpha=THOMPSON_PRIOR_ALPHA,
        prior_beta=THOMPSON_PRIOR_BETA,
    )


def make_opponent():
    """Benchmark opponent: simple UCT-MCTS with the same simulation budget."""

    return MCTSAgent.baseline_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
    )


if __name__ == "__main__":
    run_win_rate_experiment(
        experiment_name=EXPERIMENT_NAME,
        comparison_file=__file__,
        board=BOARD,
        tested_agent_factory=make_tested_agent,
        opponent_factory=make_opponent,
        seeds=SEEDS,
        output_csv=OUTPUT_CSV,
        side_mode=SIDE_MODE,
    )

"""Evaluate Enhanced UCT against the RandomAgent benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.experiments.win_rate_by_agent.benchmark_settings import (
    COLS,
    ENHANCED_UCT_HEURISTIC_WEIGHT,
    ENHANCED_USE_FORK_DETECTION,
    ROWS,
    SEEDS,
    SIDE_MODE,
    SIMULATIONS,
    UCT_C,
    WIN_LENGTH,
)
from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    run_win_rate_experiment,
)

__test__ = False

BOARD = BoardConfig(rows=ROWS, cols=COLS, win_length=WIN_LENGTH)
EXPERIMENT_NAME = "enhanced_uct_vs_random"
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_FOLDER / f"{EXPERIMENT_NAME}.csv"


def make_tested_agent() -> MCTSAgent:
    return MCTSAgent.enhanced_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
        heuristic_weight=ENHANCED_UCT_HEURISTIC_WEIGHT,
        use_fork_detection=ENHANCED_USE_FORK_DETECTION,
    )


def make_opponent() -> RandomAgent:
    return RandomAgent()


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

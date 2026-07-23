from __future__ import annotations

from common import BoardConfig, output_folder, run_win_rate_experiment
from src.agents.mcts_agent import MCTSAgent
from src.agents.tactical_agent import TacticalAgent

# Prevent pytest from treating this long-running experiment as a test file.
__test__ = False

BOARD = BoardConfig(rows=7, cols=7, win_length=4)
SIMULATIONS = 500
SEEDS = range(0, 100)
SIDE_MODE = "alternate"

EXPERIMENT_NAME = "04_enhanced_thompson_vs_tactical"
OUTPUT_CSV = output_folder() / f"{EXPERIMENT_NAME}.csv"


def make_tested_agent():
    return MCTSAgent.enhanced_thompson(simulations=SIMULATIONS)


def make_opponent():
    return TacticalAgent()


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

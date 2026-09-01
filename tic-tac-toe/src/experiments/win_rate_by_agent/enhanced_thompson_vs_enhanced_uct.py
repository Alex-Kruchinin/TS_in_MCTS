"""Run a configurable enhanced Thompson-versus-UCT experiment."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    output_folder,
    run_win_rate_experiment,
)

__test__ = False

# -----------------------------------------------------------------------
# MATCH SETTINGS
# -----------------------------------------------------------------------

ROWS = 7
COLS = 7
WIN_LENGTH = 4

# Equal simulation budget for both agents.
SIMULATIONS = 400

NUMBER_OF_GAMES = 100
SEED_START = 0
SEEDS = tuple(range(SEED_START, SEED_START + NUMBER_OF_GAMES))
SIDE_MODE = "alternate"

# Enhanced UCT parameters.
UCT_C = 0.5
UCT_HEURISTIC_WEIGHT = 0.0

# Enhanced Thompson Sampling parameters.
THOMPSON_PRIOR_ALPHA = 1.0
THOMPSON_PRIOR_BETA = 2.0


USE_FORK_DETECTION = False

# -----------------------------------------------------------------------

BOARD = BoardConfig(rows=ROWS, cols=COLS, win_length=WIN_LENGTH)

def _number_label(value: float) -> str:
    return str(value).replace(".", "_")

EXPERIMENT_NAME = (
    "enhanced_thompson_vs_enhanced_uct"
    f"_sims{SIMULATIONS}"
    f"_c{_number_label(UCT_C)}"
    f"_pb{_number_label(UCT_HEURISTIC_WEIGHT)}"
    f"_beta{_number_label(THOMPSON_PRIOR_ALPHA)}"
    f"_{_number_label(THOMPSON_PRIOR_BETA)}"
    f"_forks{str(USE_FORK_DETECTION).lower()}"
    f"_seeds{SEED_START}_{SEEDS[-1]}"
)
OUTPUT_CSV = output_folder() / f"{EXPERIMENT_NAME}.csv"


def make_tested_agent() -> MCTSAgent:
    """Enhanced Thompson; its win rate is measured in the output CSV."""

    return MCTSAgent.enhanced_thompson(
        simulations=SIMULATIONS,
        prior_alpha=THOMPSON_PRIOR_ALPHA,
        prior_beta=THOMPSON_PRIOR_BETA,
        use_fork_detection=USE_FORK_DETECTION,
    )


def make_opponent() -> MCTSAgent:
    """Enhanced UCT with explicitly configurable C and progressive bias."""

    return MCTSAgent.enhanced_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
        heuristic_weight=UCT_HEURISTIC_WEIGHT,
        use_fork_detection=USE_FORK_DETECTION,
    )


def _verify_configuration() -> None:
    thompson = make_tested_agent()
    uct = make_opponent()

    assert thompson.simulations == SIMULATIONS
    assert uct.simulations == SIMULATIONS
    assert thompson.thompson_prior_alpha == THOMPSON_PRIOR_ALPHA
    assert thompson.thompson_prior_beta == THOMPSON_PRIOR_BETA
    assert uct.selection_policy.exploration_constant == UCT_C
    assert uct.selection_policy.heuristic_weight == UCT_HEURISTIC_WEIGHT
    assert thompson.use_fork_detection is USE_FORK_DETECTION
    assert uct.use_fork_detection is USE_FORK_DETECTION


def main() -> None:
    _verify_configuration()

    print("=" * 80)
    print("ENHANCED THOMPSON VS ENHANCED UCT")
    print("=" * 80)
    print(f"Board: {ROWS}x{COLS}, k={WIN_LENGTH}")
    print(f"Games: {NUMBER_OF_GAMES}")
    print(f"Seeds: {SEEDS[0]}..{SEEDS[-1]}")
    print(f"Simulations per decision: {SIMULATIONS}")
    print(
        "Thompson prior: "
        f"Beta({THOMPSON_PRIOR_ALPHA}, {THOMPSON_PRIOR_BETA})"
    )
    print(f"UCT C: {UCT_C}")
    print(f"UCT progressive-bias weight: {UCT_HEURISTIC_WEIGHT}")
    print(f"Fork detection in both agents: {USE_FORK_DETECTION}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("=" * 80)

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


if __name__ == "__main__":
    main()

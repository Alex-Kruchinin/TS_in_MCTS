"""Run all four MCTS agents against the weak tactical benchmark."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

# Add the project root for direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.agents.weak_tactical_agent import WeakTacticalAgent
from src.experiments.win_rate_by_agent.benchmark_settings import (
    COLS,
    ENHANCED_UCT_HEURISTIC_WEIGHT,
    ENHANCED_USE_FORK_DETECTION,
    NUMBER_OF_GAMES,
    WEAK_TACTICAL_RESULTS_RUN_NAME,
    ROWS,
    SEEDS,
    SIDE_MODE,
    SIMULATIONS,
    THOMPSON_PRIOR_ALPHA,
    THOMPSON_PRIOR_BETA,
    UCT_C,
    WIN_LENGTH,
)
from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    run_win_rate_experiment,
)

# Prevent pytest from collecting this long-running experiment
__test__ = False


BOARD = BoardConfig(
    rows=ROWS,
    cols=COLS,
    win_length=WIN_LENGTH,
)

# Fork detection switch for both enhanced agents
USE_FORK_DETECTION = ENHANCED_USE_FORK_DETECTION

# Store weak-benchmark results separately
RESULTS_FOLDER = (
    Path(__file__).resolve().parent
    / "results"
    / WEAK_TACTICAL_RESULTS_RUN_NAME
)


def make_baseline_uct() -> MCTSAgent:
    """Baseline UCT using the shared benchmark C value."""
    return MCTSAgent.baseline_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
    )


def make_enhanced_uct() -> MCTSAgent:
    """Enhanced UCT using the shared benchmark C and heuristic weight."""
    return MCTSAgent.enhanced_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
        heuristic_weight=ENHANCED_UCT_HEURISTIC_WEIGHT,
        use_fork_detection=USE_FORK_DETECTION,
    )


def make_baseline_thompson() -> MCTSAgent:
    """Baseline Thompson Sampling using the shared Beta prior."""
    return MCTSAgent.baseline_thompson(
        simulations=SIMULATIONS,
        prior_alpha=THOMPSON_PRIOR_ALPHA,
        prior_beta=THOMPSON_PRIOR_BETA,
    )


def make_enhanced_thompson() -> MCTSAgent:
    """Enhanced Thompson Sampling using the shared Beta prior."""
    return MCTSAgent.enhanced_thompson(
        simulations=SIMULATIONS,
        prior_alpha=THOMPSON_PRIOR_ALPHA,
        prior_beta=THOMPSON_PRIOR_BETA,
        use_fork_detection=USE_FORK_DETECTION,
    )


def make_weak_benchmark() -> WeakTacticalAgent:
    """Return the deliberately weaker, purely rule-based benchmark."""
    return WeakTacticalAgent()


# Four agents with separate result files
EXPERIMENTS: list[tuple[str, Callable[[], MCTSAgent]]] = [
    ("baseline_uct_vs_weak_tactical", make_baseline_uct),
    ("enhanced_uct_vs_weak_tactical", make_enhanced_uct),
    ("baseline_thompson_vs_weak_tactical", make_baseline_thompson),
    ("enhanced_thompson_vs_weak_tactical", make_enhanced_thompson),
]


def _verify_fork_configuration() -> None:
    """Check that each factory uses the requested fork setting."""
    baseline_uct = make_baseline_uct()
    enhanced_uct = make_enhanced_uct()
    baseline_thompson = make_baseline_thompson()
    enhanced_thompson = make_enhanced_thompson()

    assert baseline_uct.use_fork_detection is False
    assert baseline_thompson.use_fork_detection is False
    assert enhanced_uct.use_fork_detection is USE_FORK_DETECTION
    assert enhanced_thompson.use_fork_detection is USE_FORK_DETECTION


def main() -> None:
    _verify_fork_configuration()
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALL FOUR MCTS AGENTS VS WEAK TACTICAL BENCHMARK")
    print("=" * 80)
    print(f"Board: {ROWS}x{COLS}, k={WIN_LENGTH}")
    print(f"Games per agent: {NUMBER_OF_GAMES}")
    print(f"MCTS simulations per decision: {SIMULATIONS}")
    print(f"Side mode: {SIDE_MODE}")
    print(f"UCT C: {UCT_C}")
    print(f"Fork detection in enhanced agents: {USE_FORK_DETECTION}")
    print(
        "Thompson prior: "
        f"Beta({THOMPSON_PRIOR_ALPHA}, {THOMPSON_PRIOR_BETA})"
    )
    print(f"Results folder: {RESULTS_FOLDER}")
    print(
        "WeakTacticalAgent is rule-based only: immediate win, immediate block, "
        "then centre preference. It does NOT run MCTS simulations."
    )
    print("=" * 80)

    for index, (experiment_name, agent_factory) in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        output_csv = RESULTS_FOLDER / f"{experiment_name}.csv"

        print("\n" + "#" * 80)
        print(
            f"[{index}/{len(EXPERIMENTS)}] "
            f"Starting {experiment_name}"
        )
        print("#" * 80)

        run_win_rate_experiment(
            experiment_name=experiment_name,
            comparison_file=__file__,
            board=BOARD,
            tested_agent_factory=agent_factory,
            opponent_factory=make_weak_benchmark,
            seeds=SEEDS,
            output_csv=output_csv,
            side_mode=SIDE_MODE,
        )

    print("\n" + "=" * 80)
    print("ALL FOUR WEAK-BENCHMARK EXPERIMENTS FINISHED")
    print(f"Results are in: {RESULTS_FOLDER}")
    print("=" * 80)


if __name__ == "__main__":
    main()

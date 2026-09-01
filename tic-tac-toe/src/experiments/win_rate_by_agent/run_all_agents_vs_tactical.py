"""Run all four MCTS agents against the shared tactical benchmark."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.agents.tactical_agent import TacticalAgent
from src.experiments.win_rate_by_agent.benchmark_settings import (
    COLS,
    ENHANCED_UCT_HEURISTIC_WEIGHT,
    ENHANCED_USE_FORK_DETECTION,
    NUMBER_OF_GAMES,
    TACTICAL_RESULTS_RUN_NAME,
    ROWS,
    SEEDS,
    SIDE_MODE,
    SIMULATIONS,
    TACTICAL_BENCHMARK_USE_FORK_DETECTION,
    THOMPSON_PRIOR_ALPHA,
    THOMPSON_PRIOR_BETA,
    UCT_C,
    WIN_LENGTH,
)
from src.experiments.win_rate_by_agent.common import (
    BoardConfig,
    run_win_rate_experiment,
)

__test__ = False

BOARD = BoardConfig(rows=ROWS, cols=COLS, win_length=WIN_LENGTH)

RESULTS_FOLDER = (
    Path(__file__).resolve().parent
    / "results"
    / TACTICAL_RESULTS_RUN_NAME
)


def make_baseline_uct() -> MCTSAgent:
    return MCTSAgent.baseline_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
    )


def make_enhanced_uct() -> MCTSAgent:
    return MCTSAgent.enhanced_uct(
        simulations=SIMULATIONS,
        exploration_constant=UCT_C,
        heuristic_weight=ENHANCED_UCT_HEURISTIC_WEIGHT,
        use_fork_detection=ENHANCED_USE_FORK_DETECTION,
    )


def make_baseline_thompson() -> MCTSAgent:
    return MCTSAgent.baseline_thompson(
        simulations=SIMULATIONS,
        prior_alpha=THOMPSON_PRIOR_ALPHA,
        prior_beta=THOMPSON_PRIOR_BETA,
    )


def make_enhanced_thompson() -> MCTSAgent:
    return MCTSAgent.enhanced_thompson(
        simulations=SIMULATIONS,
        prior_alpha=THOMPSON_PRIOR_ALPHA,
        prior_beta=THOMPSON_PRIOR_BETA,
        use_fork_detection=ENHANCED_USE_FORK_DETECTION,
    )


def make_benchmark() -> TacticalAgent:
    """Create the rule-based benchmark"""

    return TacticalAgent(
        use_fork_detection=TACTICAL_BENCHMARK_USE_FORK_DETECTION,
    )


EXPERIMENTS: list[tuple[str, Callable[[], MCTSAgent]]] = [
    ("baseline_uct_vs_tactical", make_baseline_uct),
    ("enhanced_uct_vs_tactical", make_enhanced_uct),
    ("baseline_thompson_vs_tactical", make_baseline_thompson),
   ("enhanced_thompson_vs_tactical", make_enhanced_thompson),
]


def _verify_configuration() -> None:
    """Check that agent factories match the shared settings"""

    baseline_uct = make_baseline_uct()
    enhanced_uct = make_enhanced_uct()
    baseline_thompson = make_baseline_thompson()
    enhanced_thompson = make_enhanced_thompson()
    benchmark = make_benchmark()

    assert baseline_uct.use_fork_detection is False
    assert baseline_thompson.use_fork_detection is False
    assert (
        enhanced_uct.use_fork_detection
        is ENHANCED_USE_FORK_DETECTION
    )
    assert (
        enhanced_thompson.use_fork_detection
        is ENHANCED_USE_FORK_DETECTION
    )
    assert not hasattr(benchmark, "simulations")
    assert (
        benchmark.use_fork_detection
        is TACTICAL_BENCHMARK_USE_FORK_DETECTION
    )


def main() -> None:
    _verify_configuration()
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALL FOUR MCTS AGENTS VS CALIBRATED TACTICAL BENCHMARK")
    print("=" * 80)
    print(f"Board: {ROWS}x{COLS}, k={WIN_LENGTH}")
    print(f"Games per agent: {NUMBER_OF_GAMES}")
    print(f"Seeds: {SEEDS[0]}..{SEEDS[-1]}")
    print(f"Side mode: {SIDE_MODE}")
    print(f"MCTS simulations per decision: {SIMULATIONS}")
    print(f"UCT C: {UCT_C}")
    print(f"Enhanced UCT heuristic weight: {ENHANCED_UCT_HEURISTIC_WEIGHT}")
    print(
        "Thompson prior: "
        f"Beta({THOMPSON_PRIOR_ALPHA}, {THOMPSON_PRIOR_BETA})"
    )
    print(
        "Fork detection in enhanced agents: "
        f"{ENHANCED_USE_FORK_DETECTION}"
    )
    print(
        "Tactical benchmark fork detection: "
        f"{TACTICAL_BENCHMARK_USE_FORK_DETECTION}"
    )
    print(
        "Benchmark remains rule-based: immediate win/block, optional fork "
        "creation/blocking, then line/centre heuristic."
    )
    print("Benchmark MCTS simulations: NONE")
    print(f"Results folder: {RESULTS_FOLDER}")
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
            opponent_factory=make_benchmark,
            seeds=SEEDS,
            output_csv=output_csv,
            side_mode=SIDE_MODE,
        )

    print("\n" + "=" * 80)
    print("ALL FOUR TACTICAL BENCHMARK EXPERIMENTS FINISHED")
    print(f"Results are in: {RESULTS_FOLDER}")
    print("=" * 80)


if __name__ == "__main__":
    main()

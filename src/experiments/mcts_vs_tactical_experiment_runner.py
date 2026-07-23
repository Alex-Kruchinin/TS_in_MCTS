from __future__ import annotations

"""
PyCharm-friendly experiment runner for MCTSAgent versus TacticalAgent.

This script is intended for harsh testing. Edit the configuration section,
right-click this file in PyCharm, and run it.

It compares several UCT-MCTS variants:
    1. baseline UCT-MCTS;
    2. UCT-MCTS with tactical guard;
    3. UCT-MCTS with tactical guard + internal heuristic rollouts;
    4. UCT-MCTS with tactical guard + heuristic expansion;
    5. enhanced UCT-MCTS with all current improvements.

The agents swap sides when SWAP_SIDES=True, which is important because
larger m,n,k games can have a strong first-player advantage.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import sys
from typing import Callable

# Allows this file to be run directly in PyCharm, even when PyCharm does not
# automatically treat the project root as a source root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_agent import Agent
from src.agents.mcts_agent import MCTSAgent
from src.agents.tactical_agent import TacticalAgent
from src.games.tic_tac_toe import Mark
from src.matches.match_runner import MatchRunner


@dataclass(frozen=True, slots=True)
class BoardConfig:
    rows: int
    cols: int
    win_length: int

    @property
    def label(self) -> str:
        return f"{self.rows}x{self.cols}_k{self.win_length}"


@dataclass(frozen=True, slots=True)
class MCTSVariant:
    name: str
    factory: Callable[[int], MCTSAgent]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    board: BoardConfig
    simulations: int
    variant_name: str
    games: int
    mcts_wins: int
    tactical_wins: int
    draws: int
    average_moves: float

    @property
    def mcts_score(self) -> float:
        return (self.mcts_wins + 0.5 * self.draws) / self.games


# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------

BOARD_CONFIGS = [
    BoardConfig(rows=7, cols=7, win_length=4),
]

SIMULATION_COUNTS = [
    200,
    1000,
]

MCTS_VARIANTS = [
    MCTSVariant(
        name="baseline_uct_random_rollout",
        factory=lambda simulations: MCTSAgent.baseline_uct(
            simulations=simulations,
        ),
    ),
    MCTSVariant(
        name="uct_tactical_guard",
        factory=lambda simulations: MCTSAgent(
            simulations=simulations,
            use_tactical_guard=True,
        ),
    ),
    MCTSVariant(
        name="uct_guard_internal_heuristic_rollout",
        factory=lambda simulations: MCTSAgent(
            simulations=simulations,
            use_tactical_guard=True,
            rollout_policy="internal_heuristic",
        ),
    ),
    MCTSVariant(
        name="uct_guard_heuristic_expansion",
        factory=lambda simulations: MCTSAgent(
            simulations=simulations,
            use_tactical_guard=True,
            use_heuristic_expansion=True,
        ),
    ),
    MCTSVariant(
        name="enhanced_uct_all_improvements",
        factory=lambda simulations: MCTSAgent.enhanced_uct(
            simulations=simulations,
        ),
    ),
]

MATCHES_PER_SETTING = 20
SWAP_SIDES = True
START_SEED = 0
SAVE_CSV = True
CSV_OUTPUT_FILE = Path(__file__).with_name("mcts_vs_tactical_results.csv")


# ---------------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------------


def play_setting(
    board: BoardConfig,
    simulations: int,
    variant: MCTSVariant,
) -> ExperimentResult:
    """Run one board/simulation/variant setting."""

    runner = MatchRunner(
        rows=board.rows,
        cols=board.cols,
        win_length=board.win_length,
    )

    mcts_wins = 0
    tactical_wins = 0
    draws = 0
    total_moves = 0

    for game_index in range(MATCHES_PER_SETTING):
        seed = START_SEED + game_index
        mcts_agent = variant.factory(simulations)
        tactical_agent = TacticalAgent()

        if SWAP_SIDES and game_index % 2 == 1:
            result = runner.play(
                agent_x=tactical_agent,
                agent_o=mcts_agent,
                seed=seed,
            )

            if result.winner == Mark.O:
                mcts_wins += 1
            elif result.winner == Mark.X:
                tactical_wins += 1
            else:
                draws += 1
        else:
            result = runner.play(
                agent_x=mcts_agent,
                agent_o=tactical_agent,
                seed=seed,
            )

            if result.winner == Mark.X:
                mcts_wins += 1
            elif result.winner == Mark.O:
                tactical_wins += 1
            else:
                draws += 1

        total_moves += result.number_of_moves

    return ExperimentResult(
        board=board,
        simulations=simulations,
        variant_name=variant.name,
        games=MATCHES_PER_SETTING,
        mcts_wins=mcts_wins,
        tactical_wins=tactical_wins,
        draws=draws,
        average_moves=total_moves / MATCHES_PER_SETTING,
    )


def run_experiments() -> list[ExperimentResult]:
    """Run all configured experiment settings."""

    results = []

    for board in BOARD_CONFIGS:
        for simulations in SIMULATION_COUNTS:
            for variant in MCTS_VARIANTS:
                result = play_setting(
                    board=board,
                    simulations=simulations,
                    variant=variant,
                )
                results.append(result)
                print_result(result)

    return results


def print_result(result: ExperimentResult) -> None:
    """Print one detailed result block."""

    print("=" * 80)
    print(f"Board:                      {result.board.label}")
    print(f"MCTS variant:               {result.variant_name}")
    print(f"MCTS simulations per move:  {result.simulations}")
    print(f"Games played:               {result.games}")
    print("-" * 80)
    print(f"MCTS wins:                  {result.mcts_wins}")
    print(f"Tactical wins:              {result.tactical_wins}")
    print(f"Draws:                      {result.draws}")
    print(f"MCTS score:                 {result.mcts_score:.3f}")
    print(f"Average game length:        {result.average_moves:.2f} moves")
    print()


def print_summary(results: list[ExperimentResult]) -> None:
    """Print a compact comparison table."""

    print("=" * 110)
    print("Summary")
    print("=" * 110)
    print(
        f"{'Board':<12} {'Sims':>6} {'Variant':<34} "
        f"{'Games':>7} {'MCTS W':>7} {'Tact W':>7} "
        f"{'Draws':>7} {'Score':>8}"
    )
    print("-" * 110)

    for result in results:
        print(
            f"{result.board.label:<12} "
            f"{result.simulations:>6} "
            f"{result.variant_name:<34} "
            f"{result.games:>7} "
            f"{result.mcts_wins:>7} "
            f"{result.tactical_wins:>7} "
            f"{result.draws:>7} "
            f"{result.mcts_score:>8.3f}"
        )


def save_csv(results: list[ExperimentResult]) -> None:
    """Save the summary results to a CSV file."""

    with CSV_OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "board",
                "rows",
                "cols",
                "win_length",
                "simulations",
                "variant",
                "games",
                "mcts_wins",
                "tactical_wins",
                "draws",
                "mcts_score",
                "average_moves",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "board": result.board.label,
                    "rows": result.board.rows,
                    "cols": result.board.cols,
                    "win_length": result.board.win_length,
                    "simulations": result.simulations,
                    "variant": result.variant_name,
                    "games": result.games,
                    "mcts_wins": result.mcts_wins,
                    "tactical_wins": result.tactical_wins,
                    "draws": result.draws,
                    "mcts_score": f"{result.mcts_score:.3f}",
                    "average_moves": f"{result.average_moves:.2f}",
                }
            )

    print(f"\nCSV saved to: {CSV_OUTPUT_FILE}")


def main() -> None:
    results = run_experiments()
    print_summary(results)

    if SAVE_CSV:
        save_csv(results)


if __name__ == "__main__":
    main()

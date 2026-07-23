from __future__ import annotations

"""
PyCharm-friendly experiment runner for comparing UCT-MCTS and
Thompson-Sampling MCTS.

Edit the configuration section, right-click this file in PyCharm, and run it.
The file swaps sides by default so that first-player advantage is controlled.
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

from src.agents.mcts_agent import MCTSAgent
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
class AgentVariant:
    name: str
    factory: Callable[[int], MCTSAgent]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    board: BoardConfig
    simulations: int
    left_variant: str
    right_variant: str
    games: int
    left_wins: int
    right_wins: int
    draws: int
    average_moves: float

    @property
    def left_score(self) -> float:
        return (self.left_wins + 0.5 * self.draws) / self.games


# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------

BOARD_CONFIGS = [
    BoardConfig(rows=3, cols=3, win_length=3),
    BoardConfig(rows=5, cols=5, win_length=4),
    BoardConfig(rows=7, cols=7, win_length=4),
]

SIMULATION_COUNTS = [
    200,
    1000,
]

# Choose one pair at a time for a very clean comparison.
# The default is the dissertation-relevant enhanced comparison.
LEFT_AGENT = AgentVariant(
    name="enhanced_uct",
    factory=lambda simulations: MCTSAgent.enhanced_uct(
        simulations=simulations,
    ),
)

RIGHT_AGENT = AgentVariant(
    name="enhanced_thompson",
    factory=lambda simulations: MCTSAgent.enhanced_thompson(
        simulations=simulations,
    ),
)

# For a baseline-only comparison, replace the two variants above with:
# LEFT_AGENT = AgentVariant("baseline_uct", MCTSAgent.baseline_uct)
# RIGHT_AGENT = AgentVariant("baseline_thompson", MCTSAgent.baseline_thompson)

MATCHES_PER_SETTING = 20
SWAP_SIDES = True
START_SEED = 0
SAVE_CSV = True
CSV_OUTPUT_FILE = Path(__file__).with_name("uct_vs_thompson_results.csv")


# ---------------------------------------------------------------------------
# Experiment logic
# ---------------------------------------------------------------------------


def play_setting(
    board: BoardConfig,
    simulations: int,
) -> ExperimentResult:
    """Run one board/simulation setting."""

    runner = MatchRunner(
        rows=board.rows,
        cols=board.cols,
        win_length=board.win_length,
    )

    left_wins = 0
    right_wins = 0
    draws = 0
    total_moves = 0

    for game_index in range(MATCHES_PER_SETTING):
        seed = START_SEED + game_index
        left_agent = LEFT_AGENT.factory(simulations)
        right_agent = RIGHT_AGENT.factory(simulations)

        if SWAP_SIDES and game_index % 2 == 1:
            result = runner.play(
                agent_x=right_agent,
                agent_o=left_agent,
                seed=seed,
            )

            if result.winner == Mark.O:
                left_wins += 1
            elif result.winner == Mark.X:
                right_wins += 1
            else:
                draws += 1
        else:
            result = runner.play(
                agent_x=left_agent,
                agent_o=right_agent,
                seed=seed,
            )

            if result.winner == Mark.X:
                left_wins += 1
            elif result.winner == Mark.O:
                right_wins += 1
            else:
                draws += 1

        total_moves += result.number_of_moves

    return ExperimentResult(
        board=board,
        simulations=simulations,
        left_variant=LEFT_AGENT.name,
        right_variant=RIGHT_AGENT.name,
        games=MATCHES_PER_SETTING,
        left_wins=left_wins,
        right_wins=right_wins,
        draws=draws,
        average_moves=total_moves / MATCHES_PER_SETTING,
    )


def run_experiments() -> list[ExperimentResult]:
    """Run all configured experiment settings."""

    results = []

    for board in BOARD_CONFIGS:
        for simulations in SIMULATION_COUNTS:
            result = play_setting(board, simulations)
            results.append(result)
            print_result(result)

    return results


def print_result(result: ExperimentResult) -> None:
    """Print one detailed result block."""

    print("=" * 80)
    print(f"Board:                      {result.board.label}")
    print(f"Left agent:                 {result.left_variant}")
    print(f"Right agent:                {result.right_variant}")
    print(f"MCTS simulations per move:  {result.simulations}")
    print(f"Games played:               {result.games}")
    print("-" * 80)
    print(f"{result.left_variant} wins: {result.left_wins}")
    print(f"{result.right_variant} wins:{result.right_wins}")
    print(f"Draws:                      {result.draws}")
    print(f"{result.left_variant} score:{result.left_score:.3f}")
    print(f"Average game length:        {result.average_moves:.2f} moves")
    print()


def print_summary(results: list[ExperimentResult]) -> None:
    """Print a compact comparison table."""

    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(
        f"{'Board':<12} {'Sims':>6} {'Left':<20} {'Right':<20} "
        f"{'Games':>7} {'L Wins':>7} {'R Wins':>7} {'Draws':>7} {'L Score':>8}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result.board.label:<12} "
            f"{result.simulations:>6} "
            f"{result.left_variant:<20} "
            f"{result.right_variant:<20} "
            f"{result.games:>7} "
            f"{result.left_wins:>7} "
            f"{result.right_wins:>7} "
            f"{result.draws:>7} "
            f"{result.left_score:>8.3f}"
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
                "left_variant",
                "right_variant",
                "games",
                "left_wins",
                "right_wins",
                "draws",
                "left_score",
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
                    "left_variant": result.left_variant,
                    "right_variant": result.right_variant,
                    "games": result.games,
                    "left_wins": result.left_wins,
                    "right_wins": result.right_wins,
                    "draws": result.draws,
                    "left_score": f"{result.left_score:.3f}",
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

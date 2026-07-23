from __future__ import annotations

"""
Comfortable experiment runner for UCT-MCTS against RandomAgent.

This file is meant to be run directly in PyCharm, for example by right-clicking
this file and choosing "Run".

It is not a pytest unit test. Unit tests check whether the code is correct.
This script runs many matches and prints experimental results.

You can change the experiment settings in the USER SETTINGS section below.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Make direct PyCharm execution reliable.
#
# If this file is run directly, Python may start inside src/experiments and may
# not know where the project root is. The following lines add the project root
# to sys.path so imports such as "from src.agents..." still work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.games.tic_tac_toe import Mark
from src.matches.match_runner import MatchRunner


@dataclass(frozen=True, slots=True)
class BoardConfig:
    """One scalable Tic-Tac-Toe board configuration."""

    rows: int
    cols: int
    win_length: int

    def label(self) -> str:
        return f"{self.rows}x{self.cols}_k{self.win_length}"


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Aggregated result for one board configuration and one MCTS budget."""

    board: BoardConfig
    simulations: int
    games_played: int
    mcts_wins: int
    random_wins: int
    draws: int
    mcts_as_x_games: int
    mcts_as_o_games: int
    total_moves: int

    @property
    def mcts_score(self) -> float:
        """Win = 1.0, draw = 0.5, loss = 0.0 from the MCTS perspective."""

        return (self.mcts_wins + 0.5 * self.draws) / self.games_played

    @property
    def average_moves(self) -> float:
        return self.total_moves / self.games_played


# =============================================================================
# USER SETTINGS
# =============================================================================

# Change these to test different board sizes and win lengths.
# Examples:
#   BoardConfig(rows=3, cols=3, win_length=3)  -> standard Tic-Tac-Toe
#   BoardConfig(rows=5, cols=5, win_length=4)  -> 5x5 board, 4 in a row wins
#   BoardConfig(rows=8, cols=4, win_length=4)  -> rectangular board
BOARD_CONFIGS = [
    BoardConfig(rows=3, cols=3, win_length=3),
    BoardConfig(rows=5, cols=5, win_length=4),
    BoardConfig(rows=8, cols=4, win_length=4),
]

# Number of MCTS simulations per real move.
# Increase these values to make MCTS stronger but slower.
SIMULATION_COUNTS = [
    50,
    200,
]

# Number of matches for each board/simulation setting.
MATCHES_PER_SETTING = 20

# If True, MCTS plays half the games as X and half as O.
# This is fairer because X moves first.
# If False, MCTS always plays as X.
SWAP_SIDES = True

# First seed used in each setting. Match n uses seed START_SEED + n.
START_SEED = 0

# Save a CSV file with the summary results.
SAVE_CSV = True
CSV_FILENAME = "mcts_vs_random_results.csv"


# =============================================================================
# EXPERIMENT CODE
# =============================================================================


def play_one_match(
    board: BoardConfig,
    simulations: int,
    seed: int,
    mcts_plays_as_x: bool,
) -> tuple[Optional[Mark], int]:
    """
    Run one match and return the winner and number of moves.

    winner is from the actual board perspective:
        Mark.X -> X won
        Mark.O -> O won
        None   -> draw
    """

    runner = MatchRunner(
        rows=board.rows,
        cols=board.cols,
        win_length=board.win_length,
    )

    mcts_agent = MCTSAgent(simulations=simulations)
    random_agent = RandomAgent()

    if mcts_plays_as_x:
        result = runner.play(
            agent_x=mcts_agent,
            agent_o=random_agent,
            seed=seed,
        )
    else:
        result = runner.play(
            agent_x=random_agent,
            agent_o=mcts_agent,
            seed=seed,
        )

    return result.winner, result.number_of_moves


def run_setting(
    board: BoardConfig,
    simulations: int,
) -> ExperimentResult:
    """Run all matches for one board configuration and one simulation budget."""

    mcts_wins = 0
    random_wins = 0
    draws = 0
    mcts_as_x_games = 0
    mcts_as_o_games = 0
    total_moves = 0

    for game_index in range(MATCHES_PER_SETTING):
        seed = START_SEED + game_index

        if SWAP_SIDES:
            mcts_plays_as_x = game_index % 2 == 0
        else:
            mcts_plays_as_x = True

        if mcts_plays_as_x:
            mcts_as_x_games += 1
        else:
            mcts_as_o_games += 1

        winner, number_of_moves = play_one_match(
            board=board,
            simulations=simulations,
            seed=seed,
            mcts_plays_as_x=mcts_plays_as_x,
        )

        total_moves += number_of_moves

        if winner is None:
            draws += 1
        elif winner == Mark.X and mcts_plays_as_x:
            mcts_wins += 1
        elif winner == Mark.O and not mcts_plays_as_x:
            mcts_wins += 1
        else:
            random_wins += 1

    return ExperimentResult(
        board=board,
        simulations=simulations,
        games_played=MATCHES_PER_SETTING,
        mcts_wins=mcts_wins,
        random_wins=random_wins,
        draws=draws,
        mcts_as_x_games=mcts_as_x_games,
        mcts_as_o_games=mcts_as_o_games,
        total_moves=total_moves,
    )


def print_result(result: ExperimentResult) -> None:
    """Print one result block in a readable format."""

    print("=" * 72)
    print(
        f"Board: {result.board.rows}x{result.board.cols}, "
        f"win length: {result.board.win_length}"
    )
    print(f"MCTS simulations per move: {result.simulations}")
    print(f"Games played:              {result.games_played}")
    print(f"MCTS as X games:           {result.mcts_as_x_games}")
    print(f"MCTS as O games:           {result.mcts_as_o_games}")
    print("-" * 72)
    print(f"MCTS wins:                 {result.mcts_wins}")
    print(f"Random wins:               {result.random_wins}")
    print(f"Draws:                     {result.draws}")
    print(f"MCTS score:                {result.mcts_score:.3f}")
    print(f"Average game length:       {result.average_moves:.2f} moves")


def save_csv(results: list[ExperimentResult]) -> Path:
    """Save all result summaries to a CSV file."""

    output_path = Path(__file__).resolve().parent / CSV_FILENAME

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rows",
                "cols",
                "win_length",
                "simulations",
                "games_played",
                "mcts_wins",
                "random_wins",
                "draws",
                "mcts_score",
                "average_moves",
                "mcts_as_x_games",
                "mcts_as_o_games",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result.board.rows,
                    result.board.cols,
                    result.board.win_length,
                    result.simulations,
                    result.games_played,
                    result.mcts_wins,
                    result.random_wins,
                    result.draws,
                    f"{result.mcts_score:.3f}",
                    f"{result.average_moves:.2f}",
                    result.mcts_as_x_games,
                    result.mcts_as_o_games,
                ]
            )

    return output_path


def main() -> None:
    """Run the full MCTS-vs-random experiment grid."""

    all_results: list[ExperimentResult] = []

    print("UCT-MCTS vs RandomAgent experiment")
    print(f"Matches per setting: {MATCHES_PER_SETTING}")
    print(f"Swap sides: {SWAP_SIDES}")
    print()

    for board in BOARD_CONFIGS:
        for simulations in SIMULATION_COUNTS:
            result = run_setting(
                board=board,
                simulations=simulations,
            )
            all_results.append(result)
            print_result(result)

    print("=" * 72)
    print("Compact summary")
    print("=" * 72)
    print(
        f"{'Board':<12} {'Sims':>8} {'Games':>8} "
        f"{'MCTS W':>8} {'Rand W':>8} {'Draws':>8} {'Score':>8}"
    )

    for result in all_results:
        print(
            f"{result.board.label():<12} "
            f"{result.simulations:>8} "
            f"{result.games_played:>8} "
            f"{result.mcts_wins:>8} "
            f"{result.random_wins:>8} "
            f"{result.draws:>8} "
            f"{result.mcts_score:>8.3f}"
        )

    if SAVE_CSV:
        output_path = save_csv(all_results)
        print()
        print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()

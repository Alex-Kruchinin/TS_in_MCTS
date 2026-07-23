from __future__ import annotations

"""
Run a small smoke experiment: UCT-MCTS as X against RandomAgent as O.

This file is intentionally an experiment script, not a pytest unit test.
It checks whether the current MCTS implementation behaves sensibly against
an easy baseline opponent.

Run from the project root with:
    python -m src.experiments.mcts_vs_random_20_matches
"""

from dataclasses import dataclass
from typing import Optional

from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.games.tic_tac_toe import Mark
from src.matches.match_runner import MatchRunner


@dataclass(frozen=True, slots=True)
class GameRecord:
    """Stores the result of one MCTS-vs-random match."""

    seed: int
    winner: Optional[Mark]
    number_of_moves: int

    @property
    def winner_name(self) -> str:
        """Return a readable winner label."""

        if self.winner == Mark.X:
            return "MCTS / X"

        if self.winner == Mark.O:
            return "Random / O"

        return "Draw"


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Aggregated results for the 20-match smoke experiment."""

    records: tuple[GameRecord, ...]
    mcts_wins: int
    random_wins: int
    draws: int

    @property
    def total_games(self) -> int:
        return len(self.records)

    @property
    def mcts_score(self) -> float:
        """
        Return score from the MCTS player's perspective.

        Scoring convention:
            win  -> 1.0
            draw -> 0.5
            loss -> 0.0
        """

        return (self.mcts_wins + 0.5 * self.draws) / self.total_games


def run_mcts_vs_random_20_matches(
    simulations: int = 200,
    number_of_matches: int = 20,
    rows: int = 3,
    cols: int = 3,
    win_length: int = 3,
) -> ExperimentSummary:
    """
    Run UCT-MCTS as X against RandomAgent as O for a fixed number of games.

    This deliberately keeps MCTS as X in every match because it is a quick
    smoke check. For a formal dissertation experiment, agents should swap
    sides to control for first-player advantage.
    """

    runner = MatchRunner(
        rows=rows,
        cols=cols,
        win_length=win_length,
    )

    mcts_agent = MCTSAgent(simulations=simulations)
    random_agent = RandomAgent()

    records: list[GameRecord] = []
    mcts_wins = 0
    random_wins = 0
    draws = 0

    for seed in range(number_of_matches):
        result = runner.play(
            agent_x=mcts_agent,
            agent_o=random_agent,
            seed=seed,
        )

        records.append(
            GameRecord(
                seed=seed,
                winner=result.winner,
                number_of_moves=result.number_of_moves,
            )
        )

        if result.winner == Mark.X:
            mcts_wins += 1
        elif result.winner == Mark.O:
            random_wins += 1
        else:
            draws += 1

    return ExperimentSummary(
        records=tuple(records),
        mcts_wins=mcts_wins,
        random_wins=random_wins,
        draws=draws,
    )


def print_summary(summary: ExperimentSummary) -> None:
    """Print per-game results and the aggregate summary."""

    print("UCT-MCTS vs RandomAgent: 20-match smoke experiment")
    print("MCTS plays as X. RandomAgent plays as O.")
    print()

    print("Per-game results:")
    print("Seed | Winner     | Moves")
    print("-----|------------|------")

    for record in summary.records:
        print(
            f"{record.seed:>4} | "
            f"{record.winner_name:<10} | "
            f"{record.number_of_moves:>5}"
        )

    print()
    print("Summary:")
    print(f"Total games: {summary.total_games}")
    print(f"MCTS wins:   {summary.mcts_wins}")
    print(f"Random wins: {summary.random_wins}")
    print(f"Draws:       {summary.draws}")
    print(f"MCTS score:  {summary.mcts_score:.3f}")


if __name__ == "__main__":
    experiment_summary = run_mcts_vs_random_20_matches()
    print_summary(experiment_summary)

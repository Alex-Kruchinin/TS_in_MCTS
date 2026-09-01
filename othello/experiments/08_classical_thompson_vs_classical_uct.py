from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import MCTSAgent, ThompsonMCTSAgent
from game import Disc, OthelloState
from matches import run_match

# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================

EXPERIMENT_NAME = "08_classical_thompson_vs_classical_uct"

# Board configuration
BOARD_ROWS = 6
BOARD_COLS = 6

# Search budget
SIMULATIONS = 400

# Use an even match count to balance colours.
TOTAL_GAMES = 100

# Swap colours and agent seeds within each reproducible pair.
SEED_START = 100000

# Thompson Sampling prior
THOMPSON_ALPHA_PRIOR = 2.0
THOMPSON_BETA_PRIOR = 1.0

# UCT exploration constant
UCT_EXPLORATION_CONSTANT = 0.5

# Console output
PRINT_EVERY_GAME = True

# Output location
OUTPUT_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME
GAME_RESULTS_CSV = OUTPUT_DIR / "game_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"


# ===========================================================================
# TIMING
# ===========================================================================

class TimedAgent:
    """Measure time spent choosing actions."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.total_seconds = 0.0
        self.decision_count = 0

    @property
    def name(self) -> str:
        return self.agent.name

    def choose_action(self, state):
        started = time.perf_counter()
        action = self.agent.choose_action(state)
        self.total_seconds += time.perf_counter() - started
        self.decision_count += 1
        return action

    @property
    def seconds_per_decision(self) -> float:
        if self.decision_count == 0:
            return 0.0
        return self.total_seconds / self.decision_count


# ===========================================================================
# RESULT STORAGE
# ===========================================================================

@dataclass
class GameRecord:
    game: int
    pair: int
    thompson_colour: str
    uct_colour: str
    outcome: str
    winner_colour: str
    thompson_discs: int
    uct_discs: int
    disc_difference: int
    actions: int
    passes: int
    match_runtime_seconds: float
    thompson_search_seconds: float
    uct_search_seconds: float
    thompson_decisions: int
    uct_decisions: int
    thompson_seconds_per_decision: float
    uct_seconds_per_decision: float
    thompson_seed: int
    uct_seed: int


# ===========================================================================
# HELPERS
# ===========================================================================

def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """95% Wilson confidence interval for a binomial proportion."""

    if total <= 0:
        return 0.0, 0.0

    p = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / total)
            + ((z * z) / (4.0 * total * total))
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def make_agents(
    thompson_seed: int,
    uct_seed: int,
) -> tuple[TimedAgent, TimedAgent]:
    """Create fresh independent agents for one match."""

    thompson = TimedAgent(
        ThompsonMCTSAgent(
            simulations=SIMULATIONS,
            alpha_prior=THOMPSON_ALPHA_PRIOR,
            beta_prior=THOMPSON_BETA_PRIOR,
            seed=thompson_seed,
            name="Classical Thompson-MCTS",
        )
    )

    uct = TimedAgent(
        MCTSAgent(
            simulations=SIMULATIONS,
            exploration_constant=UCT_EXPLORATION_CONSTANT,
            seed=uct_seed,
            name="Classical UCT-MCTS",
        )
    )

    return thompson, uct


def colour_disc_count(result, colour: Disc) -> int:
    return (
        result.black_count
        if colour is Disc.BLACK
        else result.white_count
    )


# ===========================================================================
# EXPERIMENT
# ===========================================================================

def run_experiment() -> list[GameRecord]:
    if TOTAL_GAMES <= 0:
        raise ValueError("TOTAL_GAMES must be greater than zero.")

    if TOTAL_GAMES % 2 != 0:
        raise ValueError(
            "TOTAL_GAMES must be even so Thompson and UCT receive "
            "the same number of Black and White games."
        )

    if BOARD_ROWS < 4 or BOARD_COLS < 4:
        raise ValueError("Othello board dimensions must be at least 4 x 4.")

    if BOARD_ROWS % 2 != 0 or BOARD_COLS % 2 != 0:
        raise ValueError("Othello board dimensions must both be even.")

    if SIMULATIONS <= 0:
        raise ValueError("SIMULATIONS must be greater than zero.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[GameRecord] = []

    print("=" * 72)
    print("CLASSICAL THOMPSON SAMPLING MCTS vs CLASSICAL UCT-MCTS")
    print("=" * 72)
    print(f"Board:                  {BOARD_ROWS} x {BOARD_COLS}")
    print(f"Simulations per move:   {SIMULATIONS}")
    print(f"Total games:            {TOTAL_GAMES}")
    print(
        "Thompson prior:         "
        f"Beta({THOMPSON_ALPHA_PRIOR}, {THOMPSON_BETA_PRIOR})"
    )
    print(
        "UCT exploration const.: "
        f"{UCT_EXPLORATION_CONSTANT:.6f}"
    )
    print(f"Seed start:             {SEED_START}")
    print("Colour assignment:      balanced, colour-swapped pairs")
    print("=" * 72)

    experiment_started = time.perf_counter()

    for game_index in range(TOTAL_GAMES):
        pair_index = game_index // 2
        first_seed = SEED_START + pair_index * 2
        second_seed = first_seed + 1

        # Swap colours and seeds within each pair.
        if game_index % 2 == 0:
            thompson_colour = Disc.BLACK
            uct_colour = Disc.WHITE
            thompson_seed = first_seed
            uct_seed = second_seed
        else:
            thompson_colour = Disc.WHITE
            uct_colour = Disc.BLACK
            thompson_seed = second_seed
            uct_seed = first_seed

        thompson, uct = make_agents(
            thompson_seed=thompson_seed,
            uct_seed=uct_seed,
        )

        if thompson_colour is Disc.BLACK:
            black_agent = thompson
            white_agent = uct
        else:
            black_agent = uct
            white_agent = thompson

        match_started = time.perf_counter()

        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(
                rows=BOARD_ROWS,
                cols=BOARD_COLS,
            ),
        )

        match_runtime = time.perf_counter() - match_started

        thompson_discs = colour_disc_count(result, thompson_colour)
        uct_discs = colour_disc_count(result, uct_colour)

        if result.winner is None:
            outcome = "DRAW"
            winner_colour = "DRAW"
        elif result.winner is thompson_colour:
            outcome = "THOMPSON_WIN"
            winner_colour = result.winner.name
        else:
            outcome = "UCT_WIN"
            winner_colour = result.winner.name

        record = GameRecord(
            game=game_index + 1,
            pair=pair_index + 1,
            thompson_colour=thompson_colour.name,
            uct_colour=uct_colour.name,
            outcome=outcome,
            winner_colour=winner_colour,
            thompson_discs=thompson_discs,
            uct_discs=uct_discs,
            disc_difference=thompson_discs - uct_discs,
            actions=result.action_count,
            passes=result.pass_count,
            match_runtime_seconds=match_runtime,
            thompson_search_seconds=thompson.total_seconds,
            uct_search_seconds=uct.total_seconds,
            thompson_decisions=thompson.decision_count,
            uct_decisions=uct.decision_count,
            thompson_seconds_per_decision=thompson.seconds_per_decision,
            uct_seconds_per_decision=uct.seconds_per_decision,
            thompson_seed=thompson_seed,
            uct_seed=uct_seed,
        )
        records.append(record)

        if PRINT_EVERY_GAME:
            print(
                f"Game {record.game:>3}/{TOTAL_GAMES} | "
                f"TS={record.thompson_colour:<5} | "
                f"{record.outcome:<12} | "
                f"discs {record.thompson_discs:>2}-"
                f"{record.uct_discs:<2} | "
                f"time {record.match_runtime_seconds:>7.2f}s"
            )

    total_runtime = time.perf_counter() - experiment_started

    write_game_results(records)
    summary = build_summary(records, total_runtime)
    write_summary_csv(summary)
    write_summary_text(summary)

    print_summary(summary)

    return records


# ===========================================================================
# OUTPUT
# ===========================================================================

def write_game_results(records: list[GameRecord]) -> None:
    fieldnames = list(GameRecord.__dataclass_fields__.keys())

    with GAME_RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = {
                field: getattr(record, field)
                for field in fieldnames
            }

            # Keep CSV numbers readable.
            for field in (
                "match_runtime_seconds",
                "thompson_search_seconds",
                "uct_search_seconds",
                "thompson_seconds_per_decision",
                "uct_seconds_per_decision",
            ):
                row[field] = round(float(row[field]), 6)

            writer.writerow(row)


def build_summary(
    records: list[GameRecord],
    total_runtime: float,
) -> dict[str, object]:
    total = len(records)

    thompson_wins = sum(
        record.outcome == "THOMPSON_WIN"
        for record in records
    )
    uct_wins = sum(
        record.outcome == "UCT_WIN"
        for record in records
    )
    draws = sum(
        record.outcome == "DRAW"
        for record in records
    )

    thompson_black = [
        record
        for record in records
        if record.thompson_colour == "BLACK"
    ]
    thompson_white = [
        record
        for record in records
        if record.thompson_colour == "WHITE"
    ]

    black_wins = sum(
        record.outcome == "THOMPSON_WIN"
        for record in thompson_black
    )
    white_wins = sum(
        record.outcome == "THOMPSON_WIN"
        for record in thompson_white
    )

    win_rate = thompson_wins / total
    score_rate = (thompson_wins + 0.5 * draws) / total

    black_win_rate = (
        black_wins / len(thompson_black)
        if thompson_black
        else 0.0
    )
    white_win_rate = (
        white_wins / len(thompson_white)
        if thompson_white
        else 0.0
    )

    ci_low, ci_high = wilson_interval(
        thompson_wins,
        total,
    )
    black_ci_low, black_ci_high = wilson_interval(
        black_wins,
        len(thompson_black),
    )
    white_ci_low, white_ci_high = wilson_interval(
        white_wins,
        len(thompson_white),
    )

    thompson_total_search = sum(
        record.thompson_search_seconds
        for record in records
    )
    uct_total_search = sum(
        record.uct_search_seconds
        for record in records
    )

    thompson_total_decisions = sum(
        record.thompson_decisions
        for record in records
    )
    uct_total_decisions = sum(
        record.uct_decisions
        for record in records
    )

    return {
        "experiment_name": EXPERIMENT_NAME,
        "board_rows": BOARD_ROWS,
        "board_cols": BOARD_COLS,
        "simulations_per_move": SIMULATIONS,
        "total_games": total,
        "seed_start": SEED_START,
        "thompson_alpha_prior": THOMPSON_ALPHA_PRIOR,
        "thompson_beta_prior": THOMPSON_BETA_PRIOR,
        "uct_exploration_constant": UCT_EXPLORATION_CONSTANT,

        "thompson_wins": thompson_wins,
        "uct_wins": uct_wins,
        "draws": draws,

        "thompson_win_rate": win_rate,
        "thompson_score_rate": score_rate,
        "thompson_win_rate_ci95_low": ci_low,
        "thompson_win_rate_ci95_high": ci_high,

        "thompson_black_games": len(thompson_black),
        "thompson_black_wins": black_wins,
        "thompson_black_win_rate": black_win_rate,
        "thompson_black_ci95_low": black_ci_low,
        "thompson_black_ci95_high": black_ci_high,

        "thompson_white_games": len(thompson_white),
        "thompson_white_wins": white_wins,
        "thompson_white_win_rate": white_win_rate,
        "thompson_white_ci95_low": white_ci_low,
        "thompson_white_ci95_high": white_ci_high,

        "average_thompson_discs": mean(
            [record.thompson_discs for record in records]
        ),
        "average_uct_discs": mean(
            [record.uct_discs for record in records]
        ),
        "average_disc_difference": mean(
            [record.disc_difference for record in records]
        ),
        "median_disc_difference": median(
            [record.disc_difference for record in records]
        ),

        "average_actions_per_game": mean(
            [record.actions for record in records]
        ),
        "median_actions_per_game": median(
            [record.actions for record in records]
        ),
        "average_passes_per_game": mean(
            [record.passes for record in records]
        ),

        "average_match_runtime_seconds": mean(
            [record.match_runtime_seconds for record in records]
        ),
        "total_experiment_runtime_seconds": total_runtime,

        "thompson_total_search_seconds": thompson_total_search,
        "uct_total_search_seconds": uct_total_search,

        "thompson_total_decisions": thompson_total_decisions,
        "uct_total_decisions": uct_total_decisions,

        "thompson_average_seconds_per_decision": (
            thompson_total_search / thompson_total_decisions
            if thompson_total_decisions
            else 0.0
        ),
        "uct_average_seconds_per_decision": (
            uct_total_search / uct_total_decisions
            if uct_total_decisions
            else 0.0
        ),
    }


def write_summary_csv(summary: dict[str, object]) -> None:
    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])

        for key, value in summary.items():
            if isinstance(value, float):
                writer.writerow([key, round(value, 6)])
            else:
                writer.writerow([key, value])


def write_summary_text(summary: dict[str, object]) -> None:
    text = format_summary(summary)

    with SUMMARY_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)


def format_summary(summary: dict[str, object]) -> str:
    return (
        "\n"
        "============================================================\n"
        "FINAL SUMMARY\n"
        "============================================================\n"
        f"Experiment: {summary['experiment_name']}\n"
        f"Board: {summary['board_rows']} x {summary['board_cols']}\n"
        f"Simulations per move: {summary['simulations_per_move']}\n"
        f"Games: {summary['total_games']}\n"
        "\n"
        "RESULTS\n"
        f"Thompson wins: {summary['thompson_wins']}\n"
        f"UCT wins:      {summary['uct_wins']}\n"
        f"Draws:         {summary['draws']}\n"
        "\n"
        f"Thompson win rate: "
        f"{percentage(float(summary['thompson_win_rate']))}\n"
        f"95% Wilson CI: "
        f"{percentage(float(summary['thompson_win_rate_ci95_low']))} "
        f"to "
        f"{percentage(float(summary['thompson_win_rate_ci95_high']))}\n"
        f"Thompson score rate (draw = 0.5): "
        f"{percentage(float(summary['thompson_score_rate']))}\n"
        "\n"
        "PLAYER ORDER\n"
        f"Thompson as BLACK: "
        f"{summary['thompson_black_wins']}/"
        f"{summary['thompson_black_games']} wins "
        f"({percentage(float(summary['thompson_black_win_rate']))})\n"
        f"Thompson as WHITE: "
        f"{summary['thompson_white_wins']}/"
        f"{summary['thompson_white_games']} wins "
        f"({percentage(float(summary['thompson_white_win_rate']))})\n"
        "\n"
        "GAME STATISTICS\n"
        f"Average Thompson discs: "
        f"{float(summary['average_thompson_discs']):.2f}\n"
        f"Average UCT discs:      "
        f"{float(summary['average_uct_discs']):.2f}\n"
        f"Average disc difference (TS - UCT): "
        f"{float(summary['average_disc_difference']):+.2f}\n"
        f"Average actions/game: "
        f"{float(summary['average_actions_per_game']):.2f}\n"
        f"Median actions/game: "
        f"{float(summary['median_actions_per_game']):.2f}\n"
        f"Average passes/game: "
        f"{float(summary['average_passes_per_game']):.2f}\n"
        "\n"
        "COMPUTATIONAL COST\n"
        f"Average complete match runtime: "
        f"{float(summary['average_match_runtime_seconds']):.3f} s\n"
        f"Thompson avg search time/decision: "
        f"{float(summary['thompson_average_seconds_per_decision']):.6f} s\n"
        f"UCT avg search time/decision:      "
        f"{float(summary['uct_average_seconds_per_decision']):.6f} s\n"
        f"Thompson total search time: "
        f"{float(summary['thompson_total_search_seconds']):.3f} s\n"
        f"UCT total search time:      "
        f"{float(summary['uct_total_search_seconds']):.3f} s\n"
        f"Total experiment runtime:   "
        f"{float(summary['total_experiment_runtime_seconds']):.3f} s\n"
        "============================================================\n"
    )


def print_summary(summary: dict[str, object]) -> None:
    print(format_summary(summary))
    print(f"Per-game CSV: {GAME_RESULTS_CSV}")
    print(f"Summary CSV:  {SUMMARY_CSV}")
    print(f"Summary TXT:  {SUMMARY_TXT}")


if __name__ == "__main__":
    run_experiment()

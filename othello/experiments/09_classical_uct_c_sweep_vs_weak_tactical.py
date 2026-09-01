from __future__ import annotations

import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import MCTSAgent, WeakTacticalAgent
from game import Disc, OthelloState
from matches import run_match


# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================

EXPERIMENT_NAME = "09_classical_uct_c_sweep_vs_weak_tactical"

BOARD_ROWS = 6
BOARD_COLS = 6

SIMULATIONS = 100

# Use one match per seed and an even number of seeds to balance colours
SEEDS = range(0, 100)
SIDE_MODE = "alternate"

UCT_C_VALUES = [
    0.0,
    0.2,
    0.5,
    1 / math.sqrt(2.0),
    0.75,
    1.0,
    math.sqrt(2.0),
    2.0,
    3.0,
]

# Benchmark difficulty: 0 follows its rules; 1 randomises every non-corner
WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY = 0.0

OPPONENT_SEED_OFFSET = 1_000_000

RESUME_EXISTING_RUN = True
FSYNC_AFTER_EACH_GAME = True
PRINT_EVERY_GAME = True


# ===========================================================================
# OUTPUT PATHS
# ===========================================================================

SEED_START = SEEDS.start
SEED_END = SEEDS.stop - 1

PROBABILITY_LABEL = str(
    WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY
).replace(".", "_")

RUN_NAME = (
    f"{BOARD_ROWS}x{BOARD_COLS}_"
    f"{SIMULATIONS}sims_"
    f"seed{SEED_START}-{SEED_END}_"
    f"weak_random{PROBABILITY_LABEL}"
)

RESULTS_FOLDER = (
    PROJECT_ROOT
    / "results"
    / "parameter_sweeps"
    / "uct_c"
    / RUN_NAME
)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

SWEEP_SUMMARY_CSV = RESULTS_FOLDER / "uct_c_sweep_summary.csv"


class TimedAgent:
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


def safe_c_label(c_value: float) -> str:
    if math.isclose(c_value, 1.0 / math.sqrt(2.0)):
        return "inv_sqrt2"
    if math.isclose(c_value, math.sqrt(2.0)):
        return "sqrt2"
    return str(c_value).replace(".", "_")


def percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    p = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (
        p + (z * z) / (2.0 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / total)
            + ((z * z) / (4.0 * total * total))
        )
        / denominator
    )
    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def agent_colour_for_index(index: int) -> Disc:
    if SIDE_MODE != "alternate":
        raise ValueError(
            "This script supports SIDE_MODE='alternate' only."
        )
    return Disc.BLACK if index % 2 == 0 else Disc.WHITE


def colour_disc_count(result, colour: Disc) -> int:
    if colour is Disc.BLACK:
        return result.black_count
    return result.white_count


def output_csv_for(c_value: float) -> Path:
    return RESULTS_FOLDER / f"uct_c_{safe_c_label(c_value)}.csv"


def make_mcts_agent(c_value: float, seed: int) -> TimedAgent:
    return TimedAgent(
        MCTSAgent(
            simulations=SIMULATIONS,
            exploration_constant=c_value,
            seed=seed,
            name=f"Classical UCT-MCTS (C={c_value:.6g})",
        )
    )


def make_weak_opponent(seed: int) -> TimedAgent:
    return TimedAgent(
        WeakTacticalAgent(
            random_move_probability=(
                WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY
            ),
            seed=seed + OPPONENT_SEED_OFFSET,
            name="WeakTacticalAgent",
        )
    )


FIELDNAMES = [
    "experiment_name",
    "c_value",
    "game",
    "seed",
    "tested_agent_side",
    "opponent_side",
    "outcome_for_tested_agent",
    "winner",
    "tested_agent_discs",
    "opponent_discs",
    "disc_difference",
    "moves_played",
    "passes",
    "runtime_seconds",
    "tested_search_seconds",
    "opponent_search_seconds",
    "tested_decisions",
    "opponent_decisions",
    "tested_seconds_per_decision",
    "opponent_seconds_per_decision",
    "board_rows",
    "board_cols",
    "simulations",
    "weak_tactical_random_move_probability",
]


def load_existing_rows(
    path: Path,
    c_value: float,
) -> list[dict[str, str]]:
    if not path.exists() or not RESUME_EXISTING_RUN:
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_seeds = list(SEEDS)[: len(rows)]
    actual_seeds = [int(row["seed"]) for row in rows]

    if actual_seeds != expected_seeds:
        raise RuntimeError(
            f"Cannot safely resume {path.name}: "
            "saved seeds are not the expected prefix."
        )

    for row in rows:
        if not math.isclose(
            float(row["c_value"]),
            c_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Cannot resume {path.name}: C does not match."
            )

        if int(row["simulations"]) != SIMULATIONS:
            raise RuntimeError(
                f"Cannot resume {path.name}: simulations do not match."
            )

        saved_probability = float(
            row["weak_tactical_random_move_probability"]
        )
        if not math.isclose(
            saved_probability,
            WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Cannot resume {path.name}: "
                "weak-opponent setting does not match."
            )

    return rows


def initialise_csv(
    path: Path,
    c_value: float,
) -> list[dict[str, str]]:
    existing = load_existing_rows(path, c_value)
    if existing:
        return existing

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        handle.flush()
        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    return []


def append_row(
    path: Path,
    row: dict[str, object],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writerow(row)
        handle.flush()
        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())


def current_summary(
    rows: list[dict[str, str]],
) -> dict[str, float | int]:
    completed = len(rows)
    wins = sum(
        row["outcome_for_tested_agent"] == "win"
        for row in rows
    )
    draws = sum(
        row["outcome_for_tested_agent"] == "draw"
        for row in rows
    )
    losses = sum(
        row["outcome_for_tested_agent"] == "loss"
        for row in rows
    )

    win_rate = wins / completed if completed else 0.0
    draw_rate = draws / completed if completed else 0.0
    loss_rate = losses / completed if completed else 0.0
    score_rate = (
        (wins + 0.5 * draws) / completed
        if completed
        else 0.0
    )

    black_rows = [
        row for row in rows
        if row["tested_agent_side"] == "BLACK"
    ]
    white_rows = [
        row for row in rows
        if row["tested_agent_side"] == "WHITE"
    ]

    black_wins = sum(
        row["outcome_for_tested_agent"] == "win"
        for row in black_rows
    )
    white_wins = sum(
        row["outcome_for_tested_agent"] == "win"
        for row in white_rows
    )

    ci_low, ci_high = wilson_interval(wins, completed)

    return {
        "completed": completed,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "score_rate": score_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate_as_black": (
            black_wins / len(black_rows)
            if black_rows else 0.0
        ),
        "win_rate_as_white": (
            white_wins / len(white_rows)
            if white_rows else 0.0
        ),
        "avg_runtime": mean([
            float(row["runtime_seconds"])
            for row in rows
        ]),
        "avg_disc_difference": mean([
            float(row["disc_difference"])
            for row in rows
        ]),
        "avg_moves": mean([
            float(row["moves_played"])
            for row in rows
        ]),
        "avg_tested_seconds_per_decision": mean([
            float(row["tested_seconds_per_decision"])
            for row in rows
        ]),
    }


def print_current_summary(
    rows: list[dict[str, str]],
) -> None:
    s = current_summary(rows)
    print(
        "Current summary: "
        f"completed={s['completed']}, "
        f"wins={s['wins']}, draws={s['draws']}, "
        f"losses={s['losses']}, "
        f"win_rate={percentage(float(s['win_rate']))}, "
        f"draw_rate={percentage(float(s['draw_rate']))}, "
        f"loss_rate={percentage(float(s['loss_rate']))}, "
        f"avg_runtime={float(s['avg_runtime']):.2f}s"
    )


def write_sweep_summary() -> None:
    rows_out: list[dict[str, object]] = []

    for c_value in UCT_C_VALUES:
        path = output_csv_for(c_value)
        if not path.exists():
            continue

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))

        if not rows:
            continue

        s = current_summary(rows)

        rows_out.append({
            "c_value": c_value,
            "completed_games": s["completed"],
            "wins": s["wins"],
            "draws": s["draws"],
            "losses": s["losses"],
            "win_rate": s["win_rate"],
            "score_rate": s["score_rate"],
            "ci95_low": s["ci_low"],
            "ci95_high": s["ci_high"],
            "win_rate_as_black": s["win_rate_as_black"],
            "win_rate_as_white": s["win_rate_as_white"],
            "avg_disc_difference": s["avg_disc_difference"],
            "avg_runtime_seconds": s["avg_runtime"],
            "avg_moves": s["avg_moves"],
            "avg_tested_seconds_per_decision": (
                s["avg_tested_seconds_per_decision"]
            ),
        })

    fields = [
        "c_value",
        "completed_games",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "score_rate",
        "ci95_low",
        "ci95_high",
        "win_rate_as_black",
        "win_rate_as_white",
        "avg_disc_difference",
        "avg_runtime_seconds",
        "avg_moves",
        "avg_tested_seconds_per_decision",
    ]

    temp = SWEEP_SUMMARY_CSV.with_suffix(".csv.tmp")
    with temp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
        handle.flush()
        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    os.replace(temp, SWEEP_SUMMARY_CSV)


def run_one_c_value(c_value: float) -> None:
    output_csv = output_csv_for(c_value)
    rows = initialise_csv(output_csv, c_value)

    print("\n" + "=" * 80)
    print(f"UCT exploration constant C = {c_value:.12g}")
    print(f"Output: {output_csv}")
    print("=" * 80)

    if rows:
        print(
            f"Resuming: {len(rows)}/{len(SEEDS)} games already saved."
        )
        print_current_summary(rows)

    seeds = list(SEEDS)

    for index in range(len(rows), len(seeds)):
        seed = seeds[index]
        agent_colour = agent_colour_for_index(index)
        opponent_colour = agent_colour.opponent

        if PRINT_EVERY_GAME:
            print(
                f"[{index + 1:03d}/{len(seeds):03d}] "
                f"seed={seed}, "
                f"tested_agent_as={agent_colour.name} ..."
            )

        mcts_agent = make_mcts_agent(c_value, seed)
        opponent = make_weak_opponent(seed)

        if agent_colour is Disc.BLACK:
            black_agent = mcts_agent
            white_agent = opponent
        else:
            black_agent = opponent
            white_agent = mcts_agent

        started = time.perf_counter()

        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(
                rows=BOARD_ROWS,
                cols=BOARD_COLS,
            ),
        )

        runtime = time.perf_counter() - started

        agent_discs = colour_disc_count(
            result,
            agent_colour,
        )
        opponent_discs = colour_disc_count(
            result,
            opponent_colour,
        )

        if result.winner is None:
            outcome = "draw"
            winner = "DRAW"
        elif result.winner is agent_colour:
            outcome = "win"
            winner = result.winner.name
        else:
            outcome = "loss"
            winner = result.winner.name

        row: dict[str, object] = {
            "experiment_name": EXPERIMENT_NAME,
            "c_value": c_value,
            "game": index + 1,
            "seed": seed,
            "tested_agent_side": agent_colour.name,
            "opponent_side": opponent_colour.name,
            "outcome_for_tested_agent": outcome,
            "winner": winner,
            "tested_agent_discs": agent_discs,
            "opponent_discs": opponent_discs,
            "disc_difference": agent_discs - opponent_discs,
            "moves_played": result.action_count,
            "passes": result.pass_count,
            "runtime_seconds": round(runtime, 9),
            "tested_search_seconds": round(
                mcts_agent.total_seconds,
                9,
            ),
            "opponent_search_seconds": round(
                opponent.total_seconds,
                9,
            ),
            "tested_decisions": mcts_agent.decision_count,
            "opponent_decisions": opponent.decision_count,
            "tested_seconds_per_decision": round(
                mcts_agent.seconds_per_decision,
                9,
            ),
            "opponent_seconds_per_decision": round(
                opponent.seconds_per_decision,
                9,
            ),
            "board_rows": BOARD_ROWS,
            "board_cols": BOARD_COLS,
            "simulations": SIMULATIONS,
            "weak_tactical_random_move_probability": (
                WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY
            ),
        }

        # Crash-safe: save this completed game before the next game.
        append_row(output_csv, row)
        rows.append({
            key: str(value)
            for key, value in row.items()
        })
        write_sweep_summary()

        if PRINT_EVERY_GAME:
            print(
                f"    result={outcome}, winner={winner}, "
                f"moves={result.action_count}, "
                f"runtime={runtime:.2f}s"
            )
            print_current_summary(rows)

    s = current_summary(rows)
    print(
        f"Finished C={c_value:.6g}: "
        f"wins={s['wins']}, draws={s['draws']}, "
        f"losses={s['losses']}, "
        f"win_rate={percentage(float(s['win_rate']))}, "
        f"score_rate={percentage(float(s['score_rate']))}, "
        f"avg_disc_diff="
        f"{float(s['avg_disc_difference']):+.2f}"
    )


def validate_configuration() -> None:
    if (
        BOARD_ROWS < 4
        or BOARD_COLS < 4
        or BOARD_ROWS % 2
        or BOARD_COLS % 2
    ):
        raise ValueError(
            "Othello board dimensions must be even and at least 4."
        )

    if SIMULATIONS <= 0:
        raise ValueError(
            "SIMULATIONS must be greater than zero."
        )

    if len(SEEDS) == 0:
        raise ValueError("SEEDS cannot be empty.")

    if len(SEEDS) % 2 != 0:
        raise ValueError(
            "Use an even number of seeds for balanced BLACK/WHITE games."
        )

    if any(c < 0 for c in UCT_C_VALUES):
        raise ValueError(
            "UCT exploration constants cannot be negative."
        )

    if not 0.0 <= WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY <= 1.0:
        raise ValueError(
            "WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY "
            "must be between 0 and 1."
        )


def main() -> None:
    validate_configuration()

    print("=" * 80)
    print("Othello: classical UCT C sweep vs WeakTacticalAgent")
    print(f"Board: {BOARD_ROWS}x{BOARD_COLS}")
    print(f"Simulations per MCTS move: {SIMULATIONS}")
    print(
        f"Seeds: {SEED_START}..{SEED_END} "
        f"({len(SEEDS)} games per C)"
    )
    print(
        "WeakTacticalAgent random-move probability: "
        f"{WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY:.2f}"
    )
    print(f"C values: {UCT_C_VALUES}")
    print(
        "Each completed game is appended and flushed "
        "to CSV immediately."
    )
    print(f"Results folder: {RESULTS_FOLDER}")
    print("=" * 80)

    for c_value in UCT_C_VALUES:
        run_one_c_value(c_value)

    write_sweep_summary()

    print("\nSweep complete.")
    print(f"Combined summary: {SWEEP_SUMMARY_CSV}")


if __name__ == "__main__":
    main()

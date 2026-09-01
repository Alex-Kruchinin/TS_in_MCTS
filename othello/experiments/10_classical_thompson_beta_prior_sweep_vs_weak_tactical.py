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

from agents import ThompsonMCTSAgent, WeakTacticalAgent
from game import Disc, OthelloState
from matches import run_match


# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================

EXPERIMENT_NAME = "10_classical_thompson_beta_prior_sweep_vs_weak_tactical"

BOARD_ROWS = 6
BOARD_COLS = 6

SIMULATIONS = 400

SEEDS = range(0, 100)
SIDE_MODE = "alternate"

THOMPSON_PRIORS = [
    ("beta_0_5_0_5", 0.5, 0.5),
    ("beta_1_1", 1.0, 1.0),
    ("beta_2_2", 2.0, 2.0),
    ("beta_5_5", 5.0, 5.0),
    ("beta_2_1_optimistic", 2.0, 1.0),
    ("beta_1_2_pessimistic", 1.0, 2.0),
]

WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY = 0.0

OPPONENT_SEED_OFFSET = 1_000_000

RESUME_EXISTING_RUN = True
FSYNC_AFTER_EACH_GAME = True
PRINT_EVERY_GAME = True


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
    / "thompson_beta"
    / RUN_NAME
)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

SWEEP_SUMMARY_CSV = (
    RESULTS_FOLDER / "thompson_beta_sweep_summary.csv"
)


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


def output_csv_for(label: str) -> Path:
    return RESULTS_FOLDER / f"thompson_{label}.csv"


def make_thompson_agent(
    alpha: float,
    beta: float,
    seed: int,
) -> TimedAgent:
    return TimedAgent(
        ThompsonMCTSAgent(
            simulations=SIMULATIONS,
            alpha_prior=alpha,
            beta_prior=beta,
            seed=seed,
            name=(
                "Classical Thompson-MCTS "
                f"Beta({alpha:g},{beta:g})"
            ),
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
    "prior_label",
    "alpha_prior",
    "beta_prior",
    "prior_mean",
    "prior_strength",
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
    label: str,
    alpha: float,
    beta: float,
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
        if row["prior_label"] != label:
            raise RuntimeError(
                f"Cannot resume {path.name}: label does not match."
            )

        if not math.isclose(
            float(row["alpha_prior"]),
            alpha,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Cannot resume {path.name}: alpha does not match."
            )

        if not math.isclose(
            float(row["beta_prior"]),
            beta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Cannot resume {path.name}: beta does not match."
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
    label: str,
    alpha: float,
    beta: float,
) -> list[dict[str, str]]:
    existing = load_existing_rows(
        path,
        label,
        alpha,
        beta,
    )
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

    for label, alpha, beta in THOMPSON_PRIORS:
        path = output_csv_for(label)
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
            "prior_label": label,
            "alpha_prior": alpha,
            "beta_prior": beta,
            "prior_mean": alpha / (alpha + beta),
            "prior_strength": alpha + beta,
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
        "prior_label",
        "alpha_prior",
        "beta_prior",
        "prior_mean",
        "prior_strength",
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


def run_one_prior(
    label: str,
    alpha: float,
    beta: float,
) -> None:
    output_csv = output_csv_for(label)

    rows = initialise_csv(
        output_csv,
        label,
        alpha,
        beta,
    )

    prior_mean = alpha / (alpha + beta)
    prior_strength = alpha + beta

    print("\n" + "=" * 80)
    print(
        f"Thompson prior {label}: "
        f"Beta({alpha:g}, {beta:g}), "
        f"mean={prior_mean:.3f}, "
        f"strength={prior_strength:g}"
    )
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

        thompson_agent = make_thompson_agent(
            alpha,
            beta,
            seed,
        )
        opponent = make_weak_opponent(seed)

        if agent_colour is Disc.BLACK:
            black_agent = thompson_agent
            white_agent = opponent
        else:
            black_agent = opponent
            white_agent = thompson_agent

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
            "prior_label": label,
            "alpha_prior": alpha,
            "beta_prior": beta,
            "prior_mean": prior_mean,
            "prior_strength": prior_strength,
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
                thompson_agent.total_seconds,
                9,
            ),
            "opponent_search_seconds": round(
                opponent.total_seconds,
                9,
            ),
            "tested_decisions": thompson_agent.decision_count,
            "opponent_decisions": opponent.decision_count,
            "tested_seconds_per_decision": round(
                thompson_agent.seconds_per_decision,
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
        f"Finished {label}: "
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

    for label, alpha, beta in THOMPSON_PRIORS:
        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Prior {label} must have alpha > 0 and beta > 0."
            )

    if not 0.0 <= WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY <= 1.0:
        raise ValueError(
            "WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY "
            "must be between 0 and 1."
        )


def main() -> None:
    validate_configuration()

    print("=" * 80)
    print(
        "Othello: classical Thompson Beta-prior sweep "
        "vs WeakTacticalAgent"
    )
    print(f"Board: {BOARD_ROWS}x{BOARD_COLS}")
    print(f"Simulations per MCTS move: {SIMULATIONS}")
    print(
        f"Seeds: {SEED_START}..{SEED_END} "
        f"({len(SEEDS)} games per prior)"
    )
    print(
        "WeakTacticalAgent random-move probability: "
        f"{WEAK_TACTICAL_RANDOM_MOVE_PROBABILITY:.2f}"
    )
    print("Priors:")
    for label, alpha, beta in THOMPSON_PRIORS:
        print(
            f"  {label}: Beta({alpha:g}, {beta:g}), "
            f"mean={alpha / (alpha + beta):.3f}, "
            f"strength={alpha + beta:g}"
        )
    print(
        "Each completed game is appended and flushed "
        "to CSV immediately."
    )
    print(f"Results folder: {RESULTS_FOLDER}")
    print("=" * 80)

    for label, alpha, beta in THOMPSON_PRIORS:
        run_one_prior(label, alpha, beta)

    write_sweep_summary()

    print("\nSweep complete.")
    print(f"Combined summary: {SWEEP_SUMMARY_CSV}")


if __name__ == "__main__":
    main()

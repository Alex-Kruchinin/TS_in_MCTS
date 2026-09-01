from __future__ import annotations

import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path


# ===========================================================================
# DIRECT PYCHARM EXECUTION SUPPORT
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import (
    EnhancedMCTSAgent,
    EnhancedThompsonMCTSAgent,
    MCTSAgent,
    ThompsonMCTSAgent,
)
from agents.benchmark_tactical_agent import BenchmarkTacticalAgent
from game import Disc, OthelloState
from matches import run_match


# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================


EXPERIMENT_NAME = "12_all_agents_vs_calibrated_benchmark"

RUN_VARIANTS = [
    "classical_uct",
    "classical_thompson",
    "enhanced_uct",
    "enhanced_thompson",
]

BOARD_ROWS = 6
BOARD_COLS = 6

# Uses the same budget for every MCTS agent
SIMULATIONS = 400

# Use an even game count per agent to balance colours
GAMES_PER_VARIANT = 100

# Use fresh seeds for final evaluation
SEED_START = 80_000


# ---------------------------------------------------------------------------
# Calibrated MCTS parameters
# ---------------------------------------------------------------------------

UCT_EXPLORATION_CONSTANT = 0.5

THOMPSON_ALPHA_PRIOR = 2.0
THOMPSON_BETA_PRIOR = 1.0

ENHANCED_UCT_PROGRESSIVE_BIAS_WEIGHT = 0.0

ROLLOUT_EPSILON = 0.15
EXPANSION_EPSILON = 0.05
ROLLOUT_DEPTH_LIMIT = 16

GUIDED_EXPANSION = True
HEURISTIC_ROLLOUTS = True
ROOT_SAFETY = True
ROOT_LOOKAHEAD = True
HEURISTIC_CUTOFF = True


# ---------------------------------------------------------------------------
# Intermediate rule-based benchmark
# ---------------------------------------------------------------------------

# Calibrate on pilot seeds, then freeze for final evaluation.
BENCHMARK_PROFILE = "medium"

# Higher values weaken the benchmark by choosing among its top three moves.
BENCHMARK_MISTAKE_PROBABILITY = 0.0

BENCHMARK_SEED_OFFSET = 1_000_000


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

PRINT_EVERY_GAME = True
RESUME_EXISTING_RUN = True
FSYNC_AFTER_EACH_GAME = True


# ===========================================================================
# OUTPUT
# ===========================================================================

DISPLAY_NAMES = {
    "classical_uct": "Classical UCT-MCTS",
    "classical_thompson": "Classical Thompson-MCTS",
    "enhanced_uct": "Enhanced UCT-MCTS",
    "enhanced_thompson": "Enhanced Thompson-MCTS",
}

VALID_VARIANTS = tuple(DISPLAY_NAMES)


def safe_number(value: float) -> str:
    return f"{value:g}".replace(".", "_").replace("-", "m")


RUN_NAME = (
    f"{BOARD_ROWS}x{BOARD_COLS}_"
    f"{SIMULATIONS}sims_"
    f"{GAMES_PER_VARIANT}games_each_"
    f"seed{SEED_START}_"
    f"benchmark_{BENCHMARK_PROFILE}_"
    f"mistake{safe_number(BENCHMARK_MISTAKE_PROBABILITY)}"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / EXPERIMENT_NAME
    / RUN_NAME
)

COMBINED_SUMMARY_CSV = OUTPUT_ROOT / "combined_summary.csv"
COMBINED_SUMMARY_TXT = OUTPUT_ROOT / "combined_summary.txt"


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


FIELDNAMES = [
    "experiment_name",
    "variant",
    "agent_name",
    "game",
    "seed",
    "tested_colour",
    "benchmark_colour",
    "outcome_for_tested_agent",
    "winner_colour",
    "tested_discs",
    "benchmark_discs",
    "disc_difference",
    "actions",
    "passes",
    "match_runtime_seconds",
    "tested_search_seconds",
    "benchmark_search_seconds",
    "tested_decisions",
    "benchmark_decisions",
    "tested_seconds_per_decision",
    "benchmark_seconds_per_decision",
    "tested_seed",
    "benchmark_seed",
    "board_rows",
    "board_cols",
    "simulations",
    "uct_exploration_constant",
    "thompson_alpha_prior",
    "thompson_beta_prior",
    "enhanced_uct_progressive_bias_weight",
    "benchmark_profile",
    "benchmark_mistake_probability",
]


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


def agent_colour_for_game(game_index: int) -> Disc:
    return (
        Disc.BLACK
        if game_index % 2 == 0
        else Disc.WHITE
    )


def colour_disc_count(result, colour: Disc) -> int:
    return (
        result.black_count
        if colour is Disc.BLACK
        else result.white_count
    )


def variant_folder(variant: str) -> Path:
    return OUTPUT_ROOT / variant


def game_results_path(variant: str) -> Path:
    return variant_folder(variant) / "game_results.csv"


def variant_summary_path(variant: str) -> Path:
    return variant_folder(variant) / "summary.csv"


def make_tested_agent(
    variant: str,
    seed: int,
) -> TimedAgent:
    if variant == "classical_uct":
        agent = MCTSAgent(
            simulations=SIMULATIONS,
            exploration_constant=UCT_EXPLORATION_CONSTANT,
            seed=seed,
            name=DISPLAY_NAMES[variant],
        )

    elif variant == "classical_thompson":
        agent = ThompsonMCTSAgent(
            simulations=SIMULATIONS,
            alpha_prior=THOMPSON_ALPHA_PRIOR,
            beta_prior=THOMPSON_BETA_PRIOR,
            seed=seed,
            name=DISPLAY_NAMES[variant],
        )

    elif variant == "enhanced_uct":
        agent = EnhancedMCTSAgent(
            simulations=SIMULATIONS,
            exploration_constant=UCT_EXPLORATION_CONSTANT,
            progressive_bias_weight=(
                ENHANCED_UCT_PROGRESSIVE_BIAS_WEIGHT
            ),
            rollout_epsilon=ROLLOUT_EPSILON,
            expansion_epsilon=EXPANSION_EPSILON,
            rollout_depth_limit=ROLLOUT_DEPTH_LIMIT,
            guided_expansion=GUIDED_EXPANSION,
            heuristic_rollouts=HEURISTIC_ROLLOUTS,
            root_safety=ROOT_SAFETY,
            root_lookahead=ROOT_LOOKAHEAD,
            heuristic_cutoff=HEURISTIC_CUTOFF,
            seed=seed,
            name=DISPLAY_NAMES[variant],
        )

    elif variant == "enhanced_thompson":
        agent = EnhancedThompsonMCTSAgent(
            simulations=SIMULATIONS,
            alpha_prior=THOMPSON_ALPHA_PRIOR,
            beta_prior=THOMPSON_BETA_PRIOR,
            rollout_epsilon=ROLLOUT_EPSILON,
            expansion_epsilon=EXPANSION_EPSILON,
            rollout_depth_limit=ROLLOUT_DEPTH_LIMIT,
            guided_expansion=GUIDED_EXPANSION,
            heuristic_rollouts=HEURISTIC_ROLLOUTS,
            root_safety=ROOT_SAFETY,
            root_lookahead=ROOT_LOOKAHEAD,
            heuristic_cutoff=HEURISTIC_CUTOFF,
            seed=seed,
            name=DISPLAY_NAMES[variant],
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return TimedAgent(agent)


def make_benchmark(seed: int) -> TimedAgent:
    return TimedAgent(
        BenchmarkTacticalAgent(
            profile=BENCHMARK_PROFILE,
            mistake_probability=BENCHMARK_MISTAKE_PROBABILITY,
            seed=seed + BENCHMARK_SEED_OFFSET,
            name=(
                f"BenchmarkTacticalAgent({BENCHMARK_PROFILE}, "
                f"mistake={BENCHMARK_MISTAKE_PROBABILITY:g})"
            ),
        )
    )


def initialise_csv(variant: str) -> None:
    path = game_results_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        handle.flush()

        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())


def append_row(
    variant: str,
    row: dict[str, object],
) -> None:
    with game_results_path(variant).open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writerow(row)
        handle.flush()

        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())


def load_rows(
    variant: str,
) -> list[dict[str, str]]:
    path = game_results_path(variant)

    if not RESUME_EXISTING_RUN or not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) > GAMES_PER_VARIANT:
        raise RuntimeError(
            f"{variant}: saved file contains too many games."
        )

    # Check the configuration before resuming.
    for index, row in enumerate(rows):
        expected_colour = agent_colour_for_game(index)

        checks = [
            row["variant"] == variant,
            int(row["game"]) == index + 1,
            row["tested_colour"] == expected_colour.name,
            int(row["simulations"]) == SIMULATIONS,
            int(row["board_rows"]) == BOARD_ROWS,
            int(row["board_cols"]) == BOARD_COLS,
            row["benchmark_profile"] == BENCHMARK_PROFILE,
            math.isclose(
                float(row["benchmark_mistake_probability"]),
                BENCHMARK_MISTAKE_PROBABILITY,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
        ]

        if not all(checks):
            raise RuntimeError(
                f"{variant}: existing result row {index + 1} "
                "does not match the current configuration."
            )

    return rows


def summarise(
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
        if row["tested_colour"] == "BLACK"
    ]
    white_rows = [
        row for row in rows
        if row["tested_colour"] == "WHITE"
    ]

    ci_low, ci_high = wilson_interval(
        wins,
        completed,
    )

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
        "win_rate_black": (
            sum(
                row["outcome_for_tested_agent"] == "win"
                for row in black_rows
            ) / len(black_rows)
            if black_rows
            else 0.0
        ),
        "win_rate_white": (
            sum(
                row["outcome_for_tested_agent"] == "win"
                for row in white_rows
            ) / len(white_rows)
            if white_rows
            else 0.0
        ),
        "avg_disc_difference": mean([
            float(row["disc_difference"])
            for row in rows
        ]),
        "avg_runtime": mean([
            float(row["match_runtime_seconds"])
            for row in rows
        ]),
        "avg_actions": mean([
            float(row["actions"])
            for row in rows
        ]),
        "avg_tested_seconds_per_decision": mean([
            float(row["tested_seconds_per_decision"])
            for row in rows
        ]),
    }


def print_progress(
    rows: list[dict[str, str]],
) -> None:
    s = summarise(rows)

    print(
        "Current summary: "
        f"completed={s['completed']}, "
        f"wins={s['wins']} "
        f"({percentage(float(s['win_rate']))}), "
        f"draws={s['draws']} "
        f"({percentage(float(s['draw_rate']))}), "
        f"losses={s['losses']} "
        f"({percentage(float(s['loss_rate']))}), "
        f"score_rate="
        f"{percentage(float(s['score_rate']))}, "
        f"avg_disc_diff="
        f"{float(s['avg_disc_difference']):+.2f}, "
        f"avg_runtime="
        f"{float(s['avg_runtime']):.2f}s"
    )


def write_variant_summary(
    variant: str,
    rows: list[dict[str, str]],
) -> None:
    s = summarise(rows)
    path = variant_summary_path(variant)
    temp = path.with_suffix(".csv.tmp")

    values = {
        "variant": variant,
        "agent_name": DISPLAY_NAMES[variant],
        "benchmark_profile": BENCHMARK_PROFILE,
        "benchmark_mistake_probability": (
            BENCHMARK_MISTAKE_PROBABILITY
        ),
        "completed_games": s["completed"],
        "wins": s["wins"],
        "draws": s["draws"],
        "losses": s["losses"],
        "win_rate": s["win_rate"],
        "draw_rate": s["draw_rate"],
        "loss_rate": s["loss_rate"],
        "score_rate": s["score_rate"],
        "ci95_low": s["ci_low"],
        "ci95_high": s["ci_high"],
        "win_rate_black": s["win_rate_black"],
        "win_rate_white": s["win_rate_white"],
        "avg_disc_difference": s["avg_disc_difference"],
        "avg_runtime_seconds": s["avg_runtime"],
        "avg_actions": s["avg_actions"],
        "avg_tested_seconds_per_decision": (
            s["avg_tested_seconds_per_decision"]
        ),
    }

    with temp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])

        for key, value in values.items():
            writer.writerow([key, value])

        handle.flush()

        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    os.replace(temp, path)


def write_combined_summary() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "variant",
        "agent_name",
        "completed_games",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "score_rate",
        "ci95_low",
        "ci95_high",
        "win_rate_black",
        "win_rate_white",
        "avg_disc_difference",
        "avg_runtime_seconds",
        "avg_actions",
        "avg_tested_seconds_per_decision",
        "benchmark_profile",
        "benchmark_mistake_probability",
    ]

    output_rows = []

    for variant in RUN_VARIANTS:
        path = game_results_path(variant)

        if path.exists():
            with path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = []

        s = summarise(rows)

        output_rows.append({
            "variant": variant,
            "agent_name": DISPLAY_NAMES[variant],
            "completed_games": s["completed"],
            "wins": s["wins"],
            "draws": s["draws"],
            "losses": s["losses"],
            "win_rate": s["win_rate"],
            "draw_rate": s["draw_rate"],
            "loss_rate": s["loss_rate"],
            "score_rate": s["score_rate"],
            "ci95_low": s["ci_low"],
            "ci95_high": s["ci_high"],
            "win_rate_black": s["win_rate_black"],
            "win_rate_white": s["win_rate_white"],
            "avg_disc_difference": s["avg_disc_difference"],
            "avg_runtime_seconds": s["avg_runtime"],
            "avg_actions": s["avg_actions"],
            "avg_tested_seconds_per_decision": (
                s["avg_tested_seconds_per_decision"]
            ),
            "benchmark_profile": BENCHMARK_PROFILE,
            "benchmark_mistake_probability": (
                BENCHMARK_MISTAKE_PROBABILITY
            ),
        })

    temp = COMBINED_SUMMARY_CSV.with_suffix(".csv.tmp")

    with temp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(output_rows)
        handle.flush()

        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    os.replace(
        temp,
        COMBINED_SUMMARY_CSV,
    )

    txt_temp = COMBINED_SUMMARY_TXT.with_suffix(".txt.tmp")

    with txt_temp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "ALL MCTS AGENTS VS CALIBRATED RULE-BASED BENCHMARK\n"
        )
        handle.write("=" * 80 + "\n")
        handle.write(
            f"Benchmark profile: {BENCHMARK_PROFILE}\n"
        )
        handle.write(
            "Benchmark mistake probability: "
            f"{BENCHMARK_MISTAKE_PROBABILITY}\n\n"
        )

        for row in output_rows:
            handle.write(
                f"{row['agent_name']}: "
                f"{row['wins']}W {row['draws']}D {row['losses']}L | "
                f"win={percentage(float(row['win_rate']))} | "
                f"draw={percentage(float(row['draw_rate']))} | "
                f"disc diff={float(row['avg_disc_difference']):+.2f}\n"
            )

        handle.flush()

        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    os.replace(
        txt_temp,
        COMBINED_SUMMARY_TXT,
    )


def run_variant(variant: str) -> None:
    initialise_csv(variant)
    rows = load_rows(variant)

    print("\n" + "=" * 80)
    print(
        f"{DISPLAY_NAMES[variant]} "
        f"vs BenchmarkTacticalAgent({BENCHMARK_PROFILE})"
    )
    print("=" * 80)

    if rows:
        print(
            f"Resuming: {len(rows)}/{GAMES_PER_VARIANT} "
            "games already saved."
        )
        print_progress(rows)

    for game_index in range(
        len(rows),
        GAMES_PER_VARIANT,
    ):
        seed = SEED_START + game_index
        tested_colour = agent_colour_for_game(
            game_index
        )
        benchmark_colour = tested_colour.opponent

        tested_seed = seed
        benchmark_seed = seed

        if PRINT_EVERY_GAME:
            print(
                f"[{game_index + 1:03d}/"
                f"{GAMES_PER_VARIANT:03d}] "
                f"seed={seed}, "
                f"tested_agent_as="
                f"{tested_colour.name} ..."
            )

        tested_agent = make_tested_agent(
            variant,
            tested_seed,
        )
        benchmark = make_benchmark(
            benchmark_seed,
        )

        if tested_colour is Disc.BLACK:
            black_agent = tested_agent
            white_agent = benchmark
        else:
            black_agent = benchmark
            white_agent = tested_agent

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

        tested_discs = colour_disc_count(
            result,
            tested_colour,
        )
        benchmark_discs = colour_disc_count(
            result,
            benchmark_colour,
        )

        if result.winner is None:
            outcome = "draw"
            winner = "DRAW"
        elif result.winner is tested_colour:
            outcome = "win"
            winner = result.winner.name
        else:
            outcome = "loss"
            winner = result.winner.name

        row: dict[str, object] = {
            "experiment_name": EXPERIMENT_NAME,
            "variant": variant,
            "agent_name": DISPLAY_NAMES[variant],
            "game": game_index + 1,
            "seed": seed,
            "tested_colour": tested_colour.name,
            "benchmark_colour": benchmark_colour.name,
            "outcome_for_tested_agent": outcome,
            "winner_colour": winner,
            "tested_discs": tested_discs,
            "benchmark_discs": benchmark_discs,
            "disc_difference": (
                tested_discs - benchmark_discs
            ),
            "actions": result.action_count,
            "passes": result.pass_count,
            "match_runtime_seconds": round(
                runtime,
                9,
            ),
            "tested_search_seconds": round(
                tested_agent.total_seconds,
                9,
            ),
            "benchmark_search_seconds": round(
                benchmark.total_seconds,
                9,
            ),
            "tested_decisions": (
                tested_agent.decision_count
            ),
            "benchmark_decisions": (
                benchmark.decision_count
            ),
            "tested_seconds_per_decision": round(
                tested_agent.seconds_per_decision,
                9,
            ),
            "benchmark_seconds_per_decision": round(
                benchmark.seconds_per_decision,
                9,
            ),
            "tested_seed": tested_seed,
            "benchmark_seed": benchmark_seed,
            "board_rows": BOARD_ROWS,
            "board_cols": BOARD_COLS,
            "simulations": SIMULATIONS,
            "uct_exploration_constant": (
                UCT_EXPLORATION_CONSTANT
            ),
            "thompson_alpha_prior": (
                THOMPSON_ALPHA_PRIOR
            ),
            "thompson_beta_prior": (
                THOMPSON_BETA_PRIOR
            ),
            "enhanced_uct_progressive_bias_weight": (
                ENHANCED_UCT_PROGRESSIVE_BIAS_WEIGHT
            ),
            "benchmark_profile": BENCHMARK_PROFILE,
            "benchmark_mistake_probability": (
                BENCHMARK_MISTAKE_PROBABILITY
            ),
        }

        # Save before the next game.
        append_row(
            variant,
            row,
        )

        rows.append({
            key: str(value)
            for key, value in row.items()
        })

        write_variant_summary(
            variant,
            rows,
        )
        write_combined_summary()

        if PRINT_EVERY_GAME:
            print(
                f"    result={outcome}, "
                f"winner={winner}, "
                f"discs={tested_discs}-"
                f"{benchmark_discs}, "
                f"moves={result.action_count}, "
                f"runtime={runtime:.2f}s"
            )
            print_progress(rows)
            print(
                "    saved -> "
                f"{game_results_path(variant)} "
                f"({len(rows)} completed rows)"
            )


def validate_configuration() -> None:
    unknown = [
        variant
        for variant in RUN_VARIANTS
        if variant not in VALID_VARIANTS
    ]

    if unknown:
        raise ValueError(
            f"Unknown RUN_VARIANTS: {unknown}"
        )

    if (
        BOARD_ROWS < 4
        or BOARD_COLS < 4
        or BOARD_ROWS % 2
        or BOARD_COLS % 2
    ):
        raise ValueError(
            "Board dimensions must be even and at least 4."
        )

    if SIMULATIONS <= 0:
        raise ValueError(
            "SIMULATIONS must be positive."
        )

    if (
        GAMES_PER_VARIANT <= 0
        or GAMES_PER_VARIANT % 2 != 0
    ):
        raise ValueError(
            "GAMES_PER_VARIANT must be a positive even number."
        )

    if BENCHMARK_PROFILE not in (
        "easy",
        "medium",
        "hard",
    ):
        raise ValueError(
            "BENCHMARK_PROFILE must be easy, medium, or hard."
        )

    if not 0.0 <= BENCHMARK_MISTAKE_PROBABILITY <= 1.0:
        raise ValueError(
            "BENCHMARK_MISTAKE_PROBABILITY must be in [0, 1]."
        )


def main() -> None:
    validate_configuration()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("OTHELLO AGENTS VS CALIBRATED RULE-BASED BENCHMARK")
    print("=" * 80)
    print(
        "Agents: "
        + ", ".join(
            DISPLAY_NAMES[v]
            for v in RUN_VARIANTS
        )
    )
    print(
        f"Board: {BOARD_ROWS}x{BOARD_COLS}"
    )
    print(
        f"Simulations per MCTS move: {SIMULATIONS}"
    )
    print(
        f"Games per agent: {GAMES_PER_VARIANT}"
    )
    print(
        f"Benchmark profile: {BENCHMARK_PROFILE}"
    )
    print(
        "Benchmark mistake probability: "
        f"{BENCHMARK_MISTAKE_PROBABILITY}"
    )
    print(
        "Each completed game is written and flushed "
        "before the next game starts."
    )
    print(
        f"Results: {OUTPUT_ROOT}"
    )
    print("=" * 80)

    write_combined_summary()

    for variant in RUN_VARIANTS:
        run_variant(variant)

    write_combined_summary()

    print("\nAll requested runs complete.")
    print(
        f"Combined summary: {COMBINED_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()

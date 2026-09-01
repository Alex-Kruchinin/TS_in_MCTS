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

from agents import EnhancedMCTSAgent, EnhancedThompsonMCTSAgent
from game import Disc, OthelloState
from matches import run_match


# ===========================================================================
# EXPERIMENT CONFIGURATION
# ===========================================================================

EXPERIMENT_NAME = "11_enhanced_thompson_vs_enhanced_uct"

# Othello board
BOARD_ROWS = 6
BOARD_COLS = 6

# Search budget for BOTH agents.
SIMULATIONS = 200

# Use an even count to balance colours.
TOTAL_GAMES = 100

# Swap colours and random seeds within each pair.
SEED_START = 100_000


# ---------------------------------------------------------------------------
# Parameters selected from the parameter sweeps
# ---------------------------------------------------------------------------

# Thompson Sampling prior
THOMPSON_ALPHA_PRIOR = 2.0
THOMPSON_BETA_PRIOR = 1.0

# UCT exploration constant
UCT_EXPLORATION_CONSTANT = 0.5


# ---------------------------------------------------------------------------
# Shared enhanced-MCTS settings
# ---------------------------------------------------------------------------

ROLLOUT_EPSILON = 0.15
EXPANSION_EPSILON = 0.05
ROLLOUT_DEPTH_LIMIT = 16

GUIDED_EXPANSION = True
HEURISTIC_ROLLOUTS = True
ROOT_SAFETY = True
ROOT_LOOKAHEAD = True
HEURISTIC_CUTOFF = True

# Use 0 for a selection-only comparison or 0.75 to add progressive bias.
UCT_PROGRESSIVE_BIAS_WEIGHT = 0.0


# ---------------------------------------------------------------------------
# Records in the console
# ---------------------------------------------------------------------------

PRINT_EVERY_GAME = True

# Resume completed rows after interruption.
RESUME_EXISTING_RUN = True

# Commit each completed row to disk.
FSYNC_AFTER_EACH_GAME = True


# ===========================================================================
# OUTPUT PATHS
# ===========================================================================

def safe_number(value: float) -> str:
    return f"{value:g}".replace(".", "_").replace("-", "m")


RUN_NAME = (
    f"{BOARD_ROWS}x{BOARD_COLS}_"
    f"{SIMULATIONS}sims_"
    f"{TOTAL_GAMES}games_"
    f"seed{SEED_START}_"
    f"beta{safe_number(THOMPSON_ALPHA_PRIOR)}_"
    f"{safe_number(THOMPSON_BETA_PRIOR)}_"
    f"c{safe_number(UCT_EXPLORATION_CONSTANT)}_"
    f"pb{safe_number(UCT_PROGRESSIVE_BIAS_WEIGHT)}"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / EXPERIMENT_NAME
    / RUN_NAME
)

GAME_RESULTS_CSV = OUTPUT_DIR / "game_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"


# ===========================================================================
# TIMING
# ===========================================================================

class TimedAgent:
    """Measure only the time an agent spends inside choose_action()."""

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
# CSV FIELDS
# ===========================================================================

FIELDNAMES = [
    "experiment_name",
    "game",
    "pair",
    "thompson_colour",
    "uct_colour",
    "outcome",
    "winner_colour",
    "thompson_discs",
    "uct_discs",
    "disc_difference",
    "actions",
    "passes",
    "match_runtime_seconds",
    "thompson_search_seconds",
    "uct_search_seconds",
    "thompson_decisions",
    "uct_decisions",
    "thompson_seconds_per_decision",
    "uct_seconds_per_decision",
    "thompson_seed",
    "uct_seed",
    "board_rows",
    "board_cols",
    "simulations",
    "thompson_alpha_prior",
    "thompson_beta_prior",
    "uct_exploration_constant",
    "uct_progressive_bias_weight",
    "rollout_epsilon",
    "expansion_epsilon",
    "rollout_depth_limit",
    "guided_expansion",
    "heuristic_rollouts",
    "root_safety",
    "root_lookahead",
    "heuristic_cutoff",
]




def percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """95% Wilson interval for a binomial win proportion."""

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


def colour_disc_count(result, colour: Disc) -> int:
    if colour is Disc.BLACK:
        return result.black_count
    return result.white_count


def make_agents(
    thompson_seed: int,
    uct_seed: int,
) -> tuple[TimedAgent, TimedAgent]:
    """Create fresh enhanced agents for one complete match."""

    thompson = TimedAgent(
        EnhancedThompsonMCTSAgent(
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
            seed=thompson_seed,
            name="Enhanced Thompson-MCTS",
        )
    )

    uct = TimedAgent(
        EnhancedMCTSAgent(
            simulations=SIMULATIONS,
            exploration_constant=UCT_EXPLORATION_CONSTANT,
            progressive_bias_weight=UCT_PROGRESSIVE_BIAS_WEIGHT,
            rollout_epsilon=ROLLOUT_EPSILON,
            expansion_epsilon=EXPANSION_EPSILON,
            rollout_depth_limit=ROLLOUT_DEPTH_LIMIT,
            guided_expansion=GUIDED_EXPANSION,
            heuristic_rollouts=HEURISTIC_ROLLOUTS,
            root_safety=ROOT_SAFETY,
            root_lookahead=ROOT_LOOKAHEAD,
            heuristic_cutoff=HEURISTIC_CUTOFF,
            seed=uct_seed,
            name="Enhanced UCT-MCTS",
        )
    )

    return thompson, uct


def expected_assignment(game_index: int) -> tuple[Disc, Disc, int, int]:
    """Return colours and seeds for a zero-based game index."""

    pair_index = game_index // 2
    first_seed = SEED_START + pair_index * 2
    second_seed = first_seed + 1

    if game_index % 2 == 0:
        return (
            Disc.BLACK,
            Disc.WHITE,
            first_seed,
            second_seed,
        )

    return (
        Disc.WHITE,
        Disc.BLACK,
        second_seed,
        first_seed,
    )


# ===========================================================================
# CONFIGURATION CHECK
# ===========================================================================

def validate_configuration() -> None:
    if BOARD_ROWS < 4 or BOARD_COLS < 4:
        raise ValueError(
            "Othello board dimensions must be at least 4 x 4."
        )

    if BOARD_ROWS % 2 != 0 or BOARD_COLS % 2 != 0:
        raise ValueError(
            "Othello board dimensions must both be even."
        )

    if SIMULATIONS <= 0:
        raise ValueError(
            "SIMULATIONS must be greater than zero."
        )

    if TOTAL_GAMES <= 0 or TOTAL_GAMES % 2 != 0:
        raise ValueError(
            "TOTAL_GAMES must be a positive even number."
        )

    if THOMPSON_ALPHA_PRIOR <= 0 or THOMPSON_BETA_PRIOR <= 0:
        raise ValueError(
            "Thompson alpha and beta priors must both be positive."
        )

    if UCT_EXPLORATION_CONSTANT < 0:
        raise ValueError(
            "UCT_EXPLORATION_CONSTANT cannot be negative."
        )

    if UCT_PROGRESSIVE_BIAS_WEIGHT < 0:
        raise ValueError(
            "UCT_PROGRESSIVE_BIAS_WEIGHT cannot be negative."
        )

    if not 0.0 <= ROLLOUT_EPSILON <= 1.0:
        raise ValueError(
            "ROLLOUT_EPSILON must be between 0 and 1."
        )

    if not 0.0 <= EXPANSION_EPSILON <= 1.0:
        raise ValueError(
            "EXPANSION_EPSILON must be between 0 and 1."
        )

    if ROLLOUT_DEPTH_LIMIT <= 0:
        raise ValueError(
            "ROLLOUT_DEPTH_LIMIT must be greater than zero."
        )


# ===========================================================================
# CRASH-SAFE PER-GAME RECORDING
# ===========================================================================

def initialise_results_file() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if GAME_RESULTS_CSV.exists():
        return

    with GAME_RESULTS_CSV.open(
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


def append_completed_game(row: dict[str, object]) -> None:
    """Append one completed match and commit it before the next game."""

    with GAME_RESULTS_CSV.open(
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


def load_existing_rows() -> list[dict[str, str]]:
    if (
        not RESUME_EXISTING_RUN
        or not GAME_RESULTS_CSV.exists()
    ):
        return []

    with GAME_RESULTS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) > TOTAL_GAMES:
        raise RuntimeError(
            "Existing game_results.csv contains more games than TOTAL_GAMES."
        )

    # Confirm that saved rows match this configuration.
    for index, row in enumerate(rows):
        expected_thompson_colour, expected_uct_colour, expected_ts_seed, expected_uct_seed = (
            expected_assignment(index)
        )

        checks = [
            (
                row["experiment_name"] == EXPERIMENT_NAME,
                "experiment name",
            ),
            (
                int(row["game"]) == index + 1,
                "game sequence",
            ),
            (
                row["thompson_colour"] == expected_thompson_colour.name,
                "Thompson colour assignment",
            ),
            (
                row["uct_colour"] == expected_uct_colour.name,
                "UCT colour assignment",
            ),
            (
                int(row["thompson_seed"]) == expected_ts_seed,
                "Thompson seed",
            ),
            (
                int(row["uct_seed"]) == expected_uct_seed,
                "UCT seed",
            ),
            (
                int(row["board_rows"]) == BOARD_ROWS,
                "board rows",
            ),
            (
                int(row["board_cols"]) == BOARD_COLS,
                "board columns",
            ),
            (
                int(row["simulations"]) == SIMULATIONS,
                "simulation budget",
            ),
            (
                math.isclose(
                    float(row["thompson_alpha_prior"]),
                    THOMPSON_ALPHA_PRIOR,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "Thompson alpha prior",
            ),
            (
                math.isclose(
                    float(row["thompson_beta_prior"]),
                    THOMPSON_BETA_PRIOR,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "Thompson beta prior",
            ),
            (
                math.isclose(
                    float(row["uct_exploration_constant"]),
                    UCT_EXPLORATION_CONSTANT,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "UCT exploration constant",
            ),
            (
                math.isclose(
                    float(row["uct_progressive_bias_weight"]),
                    UCT_PROGRESSIVE_BIAS_WEIGHT,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "UCT progressive-bias weight",
            ),
        ]

        failed = [
            description
            for passed, description in checks
            if not passed
        ]

        if failed:
            raise RuntimeError(
                "Cannot safely resume the existing experiment because "
                f"game {index + 1} does not match the current configuration: "
                + ", ".join(failed)
            )

    return rows


# ===========================================================================
# SUMMARY
# ===========================================================================

def build_summary(
    rows: list[dict[str, str]],
    *,
    elapsed_this_session: float = 0.0,
) -> dict[str, object]:
    total = len(rows)

    thompson_wins = sum(
        row["outcome"] == "THOMPSON_WIN"
        for row in rows
    )
    uct_wins = sum(
        row["outcome"] == "UCT_WIN"
        for row in rows
    )
    draws = sum(
        row["outcome"] == "DRAW"
        for row in rows
    )

    thompson_score_rate = (
        (thompson_wins + 0.5 * draws) / total
        if total
        else 0.0
    )
    uct_score_rate = (
        (uct_wins + 0.5 * draws) / total
        if total
        else 0.0
    )

    thompson_win_rate = (
        thompson_wins / total
        if total
        else 0.0
    )
    uct_win_rate = (
        uct_wins / total
        if total
        else 0.0
    )
    draw_rate = (
        draws / total
        if total
        else 0.0
    )

    thompson_black = [
        row
        for row in rows
        if row["thompson_colour"] == "BLACK"
    ]
    thompson_white = [
        row
        for row in rows
        if row["thompson_colour"] == "WHITE"
    ]

    thompson_black_wins = sum(
        row["outcome"] == "THOMPSON_WIN"
        for row in thompson_black
    )
    thompson_white_wins = sum(
        row["outcome"] == "THOMPSON_WIN"
        for row in thompson_white
    )

    ts_ci_low, ts_ci_high = wilson_interval(
        thompson_wins,
        total,
    )
    uct_ci_low, uct_ci_high = wilson_interval(
        uct_wins,
        total,
    )

    match_runtimes = [
        float(row["match_runtime_seconds"])
        for row in rows
    ]
    disc_differences = [
        float(row["disc_difference"])
        for row in rows
    ]
    actions = [
        float(row["actions"])
        for row in rows
    ]
    passes = [
        float(row["passes"])
        for row in rows
    ]

    thompson_total_search = sum(
        float(row["thompson_search_seconds"])
        for row in rows
    )
    uct_total_search = sum(
        float(row["uct_search_seconds"])
        for row in rows
    )

    thompson_total_decisions = sum(
        int(row["thompson_decisions"])
        for row in rows
    )
    uct_total_decisions = sum(
        int(row["uct_decisions"])
        for row in rows
    )

    return {
        "experiment_name": EXPERIMENT_NAME,
        "board_rows": BOARD_ROWS,
        "board_cols": BOARD_COLS,
        "simulations_per_move": SIMULATIONS,
        "planned_games": TOTAL_GAMES,
        "completed_games": total,
        "seed_start": SEED_START,

        "thompson_alpha_prior": THOMPSON_ALPHA_PRIOR,
        "thompson_beta_prior": THOMPSON_BETA_PRIOR,
        "uct_exploration_constant": UCT_EXPLORATION_CONSTANT,
        "uct_progressive_bias_weight": UCT_PROGRESSIVE_BIAS_WEIGHT,

        "rollout_epsilon": ROLLOUT_EPSILON,
        "expansion_epsilon": EXPANSION_EPSILON,
        "rollout_depth_limit": ROLLOUT_DEPTH_LIMIT,
        "guided_expansion": GUIDED_EXPANSION,
        "heuristic_rollouts": HEURISTIC_ROLLOUTS,
        "root_safety": ROOT_SAFETY,
        "root_lookahead": ROOT_LOOKAHEAD,
        "heuristic_cutoff": HEURISTIC_CUTOFF,

        "thompson_wins": thompson_wins,
        "uct_wins": uct_wins,
        "draws": draws,

        "thompson_win_rate": thompson_win_rate,
        "uct_win_rate": uct_win_rate,
        "draw_rate": draw_rate,

        "thompson_score_rate": thompson_score_rate,
        "uct_score_rate": uct_score_rate,

        "thompson_win_rate_ci95_low": ts_ci_low,
        "thompson_win_rate_ci95_high": ts_ci_high,
        "uct_win_rate_ci95_low": uct_ci_low,
        "uct_win_rate_ci95_high": uct_ci_high,

        "thompson_black_games": len(thompson_black),
        "thompson_black_wins": thompson_black_wins,
        "thompson_black_win_rate": (
            thompson_black_wins / len(thompson_black)
            if thompson_black
            else 0.0
        ),

        "thompson_white_games": len(thompson_white),
        "thompson_white_wins": thompson_white_wins,
        "thompson_white_win_rate": (
            thompson_white_wins / len(thompson_white)
            if thompson_white
            else 0.0
        ),

        "average_thompson_discs": mean([
            float(row["thompson_discs"])
            for row in rows
        ]),
        "average_uct_discs": mean([
            float(row["uct_discs"])
            for row in rows
        ]),
        "average_disc_difference_ts_minus_uct": (
            mean(disc_differences)
        ),
        "median_disc_difference_ts_minus_uct": (
            median(disc_differences)
        ),

        "average_actions_per_game": mean(actions),
        "median_actions_per_game": median(actions),
        "average_passes_per_game": mean(passes),

        "average_match_runtime_seconds": mean(match_runtimes),
        "recorded_match_runtime_seconds": sum(match_runtimes),
        "elapsed_this_session_seconds": elapsed_this_session,

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


def format_summary(summary: dict[str, object]) -> str:
    return (
        "\n"
        "============================================================\n"
        "CURRENT / FINAL SUMMARY\n"
        "============================================================\n"
        f"Experiment: {summary['experiment_name']}\n"
        f"Board: {summary['board_rows']} x {summary['board_cols']}\n"
        f"Simulations per move: {summary['simulations_per_move']}\n"
        f"Games: {summary['completed_games']}/"
        f"{summary['planned_games']}\n"
        "\n"
        "RESULTS\n"
        f"Enhanced Thompson wins: {summary['thompson_wins']}\n"
        f"Enhanced UCT wins:      {summary['uct_wins']}\n"
        f"Draws:                  {summary['draws']}\n"
        "\n"
        f"Enhanced Thompson win rate: "
        f"{percentage(float(summary['thompson_win_rate']))}\n"
        f"Enhanced UCT win rate:      "
        f"{percentage(float(summary['uct_win_rate']))}\n"
        f"Draw rate:                  "
        f"{percentage(float(summary['draw_rate']))}\n"
        "\n"
        f"Enhanced Thompson score rate: "
        f"{percentage(float(summary['thompson_score_rate']))}\n"
        f"Enhanced UCT score rate:      "
        f"{percentage(float(summary['uct_score_rate']))}\n"
        "\n"
        f"Thompson 95% Wilson win CI: "
        f"{percentage(float(summary['thompson_win_rate_ci95_low']))} "
        f"to "
        f"{percentage(float(summary['thompson_win_rate_ci95_high']))}\n"
        f"UCT 95% Wilson win CI:      "
        f"{percentage(float(summary['uct_win_rate_ci95_low']))} "
        f"to "
        f"{percentage(float(summary['uct_win_rate_ci95_high']))}\n"
        "\n"
        "THOMPSON PLAYER ORDER\n"
        f"As BLACK: {summary['thompson_black_wins']}/"
        f"{summary['thompson_black_games']} wins "
        f"({percentage(float(summary['thompson_black_win_rate']))})\n"
        f"As WHITE: {summary['thompson_white_wins']}/"
        f"{summary['thompson_white_games']} wins "
        f"({percentage(float(summary['thompson_white_win_rate']))})\n"
        "\n"
        "GAME STATISTICS\n"
        f"Average Thompson discs: "
        f"{float(summary['average_thompson_discs']):.2f}\n"
        f"Average UCT discs:      "
        f"{float(summary['average_uct_discs']):.2f}\n"
        f"Average disc difference (TS - UCT): "
        f"{float(summary['average_disc_difference_ts_minus_uct']):+.2f}\n"
        f"Average actions/game: "
        f"{float(summary['average_actions_per_game']):.2f}\n"
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
        "============================================================\n"
    )


def write_summary_files(
    rows: list[dict[str, str]],
    *,
    elapsed_this_session: float,
) -> dict[str, object]:
    """Rewrite summary CSV/TXT after every completed game."""

    summary = build_summary(
        rows,
        elapsed_this_session=elapsed_this_session,
    )

    # Replace summaries only after temporary files are complete.
    summary_csv_tmp = SUMMARY_CSV.with_suffix(".csv.tmp")
    with summary_csv_tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])

        for key, value in summary.items():
            if isinstance(value, float):
                writer.writerow([key, round(value, 9)])
            else:
                writer.writerow([key, value])

        handle.flush()
        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    os.replace(summary_csv_tmp, SUMMARY_CSV)

    summary_txt_tmp = SUMMARY_TXT.with_suffix(".txt.tmp")
    with summary_txt_tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(format_summary(summary))
        handle.flush()

        if FSYNC_AFTER_EACH_GAME:
            os.fsync(handle.fileno())

    os.replace(summary_txt_tmp, SUMMARY_TXT)

    return summary


def print_current_rates(
    rows: list[dict[str, str]],
    *,
    elapsed_this_session: float,
) -> None:
    summary = build_summary(
        rows,
        elapsed_this_session=elapsed_this_session,
    )

    print(
        "Current summary: "
        f"completed={summary['completed_games']}, "
        f"Thompson wins={summary['thompson_wins']} "
        f"({percentage(float(summary['thompson_win_rate']))}), "
        f"UCT wins={summary['uct_wins']} "
        f"({percentage(float(summary['uct_win_rate']))}), "
        f"draws={summary['draws']} "
        f"({percentage(float(summary['draw_rate']))}), "
        f"TS score={percentage(float(summary['thompson_score_rate']))}, "
        f"UCT score={percentage(float(summary['uct_score_rate']))}, "
        f"avg_runtime="
        f"{float(summary['average_match_runtime_seconds']):.2f}s"
    )


# ===========================================================================
# EXPERIMENT
# ===========================================================================

def run_experiment() -> None:
    validate_configuration()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    initialise_results_file()

    rows = load_existing_rows()

    print("=" * 80)
    print("ENHANCED THOMPSON-MCTS vs ENHANCED UCT-MCTS")
    print("=" * 80)
    print(f"Board:                       {BOARD_ROWS} x {BOARD_COLS}")
    print(f"Simulations per move:        {SIMULATIONS}")
    print(f"Total games:                 {TOTAL_GAMES}")
    print(
        "Thompson prior:              "
        f"Beta({THOMPSON_ALPHA_PRIOR}, {THOMPSON_BETA_PRIOR})"
    )
    print(
        "UCT exploration constant:    "
        f"{UCT_EXPLORATION_CONSTANT}"
    )
    print(
        "UCT progressive-bias weight: "
        f"{UCT_PROGRESSIVE_BIAS_WEIGHT}"
    )
    print(f"Seed start:                  {SEED_START}")
    print("Colour assignment:           balanced colour-swapped pairs")
    print("Guided expansion:            ", GUIDED_EXPANSION)
    print("Heuristic rollouts:          ", HEURISTIC_ROLLOUTS)
    print("Root safety:                 ", ROOT_SAFETY)
    print("Root lookahead:              ", ROOT_LOOKAHEAD)
    print("Heuristic cutoff:            ", HEURISTIC_CUTOFF)
    print(f"Results folder:              {OUTPUT_DIR}")
    print("=" * 80)

    if rows:
        print(
            f"Resuming existing experiment: "
            f"{len(rows)}/{TOTAL_GAMES} games already saved."
        )
        print_current_rates(
            rows,
            elapsed_this_session=0.0,
        )

    session_started = time.perf_counter()

    for game_index in range(len(rows), TOTAL_GAMES):
        pair_index = game_index // 2

        (
            thompson_colour,
            uct_colour,
            thompson_seed,
            uct_seed,
        ) = expected_assignment(game_index)

        if PRINT_EVERY_GAME:
            print(
                f"[{game_index + 1:03d}/{TOTAL_GAMES:03d}] "
                f"pair={pair_index + 1}, "
                f"Thompson={thompson_colour.name}, "
                f"UCT={uct_colour.name} ..."
            )

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

        thompson_discs = colour_disc_count(
            result,
            thompson_colour,
        )
        uct_discs = colour_disc_count(
            result,
            uct_colour,
        )

        if result.winner is None:
            outcome = "DRAW"
            winner_colour = "DRAW"
        elif result.winner is thompson_colour:
            outcome = "THOMPSON_WIN"
            winner_colour = result.winner.name
        else:
            outcome = "UCT_WIN"
            winner_colour = result.winner.name

        row: dict[str, object] = {
            "experiment_name": EXPERIMENT_NAME,
            "game": game_index + 1,
            "pair": pair_index + 1,
            "thompson_colour": thompson_colour.name,
            "uct_colour": uct_colour.name,
            "outcome": outcome,
            "winner_colour": winner_colour,
            "thompson_discs": thompson_discs,
            "uct_discs": uct_discs,
            "disc_difference": thompson_discs - uct_discs,
            "actions": result.action_count,
            "passes": result.pass_count,
            "match_runtime_seconds": round(match_runtime, 9),
            "thompson_search_seconds": round(
                thompson.total_seconds,
                9,
            ),
            "uct_search_seconds": round(
                uct.total_seconds,
                9,
            ),
            "thompson_decisions": thompson.decision_count,
            "uct_decisions": uct.decision_count,
            "thompson_seconds_per_decision": round(
                thompson.seconds_per_decision,
                9,
            ),
            "uct_seconds_per_decision": round(
                uct.seconds_per_decision,
                9,
            ),
            "thompson_seed": thompson_seed,
            "uct_seed": uct_seed,

            "board_rows": BOARD_ROWS,
            "board_cols": BOARD_COLS,
            "simulations": SIMULATIONS,

            "thompson_alpha_prior": THOMPSON_ALPHA_PRIOR,
            "thompson_beta_prior": THOMPSON_BETA_PRIOR,
            "uct_exploration_constant": UCT_EXPLORATION_CONSTANT,
            "uct_progressive_bias_weight": UCT_PROGRESSIVE_BIAS_WEIGHT,

            "rollout_epsilon": ROLLOUT_EPSILON,
            "expansion_epsilon": EXPANSION_EPSILON,
            "rollout_depth_limit": ROLLOUT_DEPTH_LIMIT,
            "guided_expansion": GUIDED_EXPANSION,
            "heuristic_rollouts": HEURISTIC_ROLLOUTS,
            "root_safety": ROOT_SAFETY,
            "root_lookahead": ROOT_LOOKAHEAD,
            "heuristic_cutoff": HEURISTIC_CUTOFF,
        }

        # Save this game before starting the next one.
        append_completed_game(row)

        # Match the string format returned by csv.DictReader.
        rows.append({
            key: str(value)
            for key, value in row.items()
        })

        elapsed_this_session = (
            time.perf_counter() - session_started
        )

        # Rewrite summaries after every game.
        write_summary_files(
            rows,
            elapsed_this_session=elapsed_this_session,
        )

        if PRINT_EVERY_GAME:
            print(
                f"    result={outcome}, winner={winner_colour}, "
                f"discs={thompson_discs}-{uct_discs}, "
                f"moves={result.action_count}, "
                f"runtime={match_runtime:.2f}s"
            )
            print_current_rates(
                rows,
                elapsed_this_session=elapsed_this_session,
            )
            print(
                f"    saved -> {GAME_RESULTS_CSV.name} "
                f"({len(rows)} completed rows)"
            )

    elapsed_this_session = (
        time.perf_counter() - session_started
    )

    final_summary = write_summary_files(
        rows,
        elapsed_this_session=elapsed_this_session,
    )

    print(format_summary(final_summary))
    print(f"Per-game CSV: {GAME_RESULTS_CSV}")
    print(f"Summary CSV:  {SUMMARY_CSV}")
    print(f"Summary TXT:  {SUMMARY_TXT}")


if __name__ == "__main__":
    run_experiment()

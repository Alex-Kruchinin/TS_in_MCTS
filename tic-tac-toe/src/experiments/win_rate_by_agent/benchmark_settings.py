"""Shared settings for the main Tic-Tac-Toe benchmarks."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------

ROWS = 7
COLS = 7
WIN_LENGTH = 4

# MCTS simulation budget per move
SIMULATIONS = 400

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

NUMBER_OF_GAMES = 100
SIDE_MODE = "alternate"

# First random seed used in the experiment
SEED_START = 10000

SEEDS = tuple(range(SEED_START, SEED_START + NUMBER_OF_GAMES))



# ---------------------------------------------------------------------------
# UCT configuration
# ---------------------------------------------------------------------------

# Used by BOTH baseline UCT and enhanced UCT
UCT_C = 0.5

# Progressive-bias weight used only by enhanced UCT selection (Was set to 0 in all experiments)
ENHANCED_UCT_HEURISTIC_WEIGHT = 0.0



# ---------------------------------------------------------------------------
# Thompson Sampling configuration
# ---------------------------------------------------------------------------

# Used by BOTH baseline Thompson and enhanced Thompson.
THOMPSON_PRIOR_ALPHA = 1.0
THOMPSON_PRIOR_BETA = 2.0


# ---------------------------------------------------------------------------
# Enhanced-agent / benchmark fork detection
# ---------------------------------------------------------------------------

# Fork detection switch for both enhanced agents
ENHANCED_USE_FORK_DETECTION = False

# Optional fork handling for the rule-based benchmark
TACTICAL_BENCHMARK_USE_FORK_DETECTION = True

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Include seed in the folder name
SEED_END = SEEDS[-1] if SEEDS else SEED_START
RESULTS_RUN_NAME = (
    f"final_benchmark_optimized_seeds_{SEED_START}_{SEED_END}"
)

FORK_MODE_NAME = (
    "with_forks" if ENHANCED_USE_FORK_DETECTION else "no_forks"
)
TACTICAL_BENCHMARK_MODE_NAME = (
    "tactical_benchmarkforks_"
    f"{TACTICAL_BENCHMARK_USE_FORK_DETECTION}_"
    f"mcts_{FORK_MODE_NAME}"
).lower()
TACTICAL_RESULTS_RUN_NAME = (
    f"{RESULTS_RUN_NAME}_{TACTICAL_BENCHMARK_MODE_NAME}"
)
WEAK_TACTICAL_RESULTS_RUN_NAME = (
    f"{RESULTS_RUN_NAME}_weak_tactical_{FORK_MODE_NAME}"
)

from __future__ import annotations

"""
PyCharm-friendly script for displaying one complete match trace.

This file is intended for inspection and dissertation demonstration rather
than large statistical evaluation. Edit the configuration section below,
right-click this file in PyCharm, and run it.
"""

from pathlib import Path
import sys

# Add the project root for direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mcts_agent import MCTSAgent
from src.agents.tactical_agent import TacticalAgent
from src.matches.match_runner import MatchRunner


# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------

ROWS = 7
COLS = 7
WIN_LENGTH = 4

SEED = 42

AGENT_X = MCTSAgent.enhanced_thompson(simulations=500)
AGENT_O = TacticalAgent()

SAVE_TRACE_TO_FILE = True
TRACE_OUTPUT_FILE = Path(__file__).with_name("single_match_trace.txt")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    """Run one match and print every board state until termination."""

    runner = MatchRunner(
        rows=ROWS,
        cols=COLS,
        win_length=WIN_LENGTH,
    )

    result = runner.play(
        agent_x=AGENT_X,
        agent_o=AGENT_O,
        seed=SEED,
        record_trace=True,
    )

    if result.trace is None:
        raise RuntimeError("Trace was not recorded.")

    trace_text = result.trace.to_text()
    print(trace_text)

    if SAVE_TRACE_TO_FILE:
        TRACE_OUTPUT_FILE.write_text(trace_text, encoding="utf-8")
        print(f"\nTrace saved to: {TRACE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.tic_tac_toe import Mark
from src.matches.match_runner import MatchRunner


@dataclass(frozen=True, slots=True)
class BoardConfig:
    rows: int
    cols: int
    win_length: int


AgentFactory = Callable[[], object]


# CSV columns used for results, plotting and resumed runs
FIELDNAMES = [
    "status",
    "experiment_name",
    "comparison_file",
    "game_index",
    "seed",
    "board_rows",
    "board_cols",
    "win_length",
    "tested_agent_name",
    "opponent_name",
    "tested_agent_side",
    "opponent_side",
    "agent_x_name",
    "agent_o_name",
    "winner",
    "outcome_for_tested_agent",
    "moves_played",
    "runtime_seconds",
    "seconds_per_move",
    "timestamp_start",
    "timestamp_end",
    "tested_agent_simulations",
    "tested_agent_selection_policy",
    "tested_agent_exploration_constant",
    "tested_agent_heuristic_weight",
    "tested_agent_thompson_prior_alpha",
    "tested_agent_thompson_prior_beta",
    "tested_agent_rollout_policy",
    "tested_agent_use_tactical_guard",
    "tested_agent_use_heuristic_expansion",
    "tested_agent_use_fork_detection",
    "opponent_simulations",
    "opponent_selection_policy",
    "opponent_exploration_constant",
    "opponent_heuristic_weight",
    "opponent_thompson_prior_alpha",
    "opponent_thompson_prior_beta",
    "opponent_rollout_policy",
    "opponent_use_tactical_guard",
    "opponent_use_heuristic_expansion",
    "opponent_use_fork_detection",
    "error_message",
]

# Output folder
def output_folder() -> Path:
    folder = Path(__file__).resolve().parent / "results"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def display_agent_name(agent: object) -> str:
    """Return a readable name, including key MCTS options if available."""

    class_name = agent.__class__.__name__
    simulations = getattr(agent, "simulations", None)

    if simulations is None:
        custom_text = str(agent)
        default_prefix = f"<{agent.__class__.__module__}."
        if not custom_text.startswith(default_prefix):
            return custom_text
        return class_name

    selection_policy = getattr(agent, "selection_policy", None)
    exploration_constant = getattr(selection_policy, "exploration_constant", None)
    heuristic_weight = getattr(selection_policy, "heuristic_weight", None)
    prior_alpha = getattr(agent, "thompson_prior_alpha", None)
    prior_beta = getattr(agent, "thompson_prior_beta", None)
    rollout_agent = getattr(agent, "rollout_agent", None)
    rollout_policy = getattr(agent, "rollout_policy", None)
    tactical_guard = getattr(agent, "use_tactical_guard", None)
    heuristic_expansion = getattr(agent, "use_heuristic_expansion", None)
    fork_detection = getattr(agent, "use_fork_detection", None)

    selection_name = (
        selection_policy.__class__.__name__
        if selection_policy is not None
        else "unknown"
    )
    if hasattr(agent, "rollout_policy_name"):
        rollout_name = agent.rollout_policy_name()
    elif rollout_agent is not None:
        rollout_name = rollout_agent.__class__.__name__
    elif rollout_policy is not None:
        rollout_name = str(rollout_policy)
    else:
        rollout_name = "unknown"

    extra_parts = []
    if exploration_constant is not None:
        extra_parts.append(f"C={exploration_constant:.4g}")
    if heuristic_weight is not None:
        extra_parts.append(f"heuristic_weight={heuristic_weight:.4g}")
    if selection_name == "ThompsonSamplingSelectionPolicy":
        extra_parts.append(f"prior=Beta({prior_alpha:g}, {prior_beta:g})")
    extra = ", " + ", ".join(extra_parts) if extra_parts else ""

    return (
        f"{class_name}(sim={simulations}, "
        f"selection={selection_name}{extra}, "
        f"rollout={rollout_name}, "
        f"guard={tactical_guard}, "
        f"heuristic_expansion={heuristic_expansion}, "
        f"fork_detection={fork_detection})"
    )


def agent_config(agent: object, prefix: str) -> dict[str, object]:
    selection_policy = getattr(agent, "selection_policy", None)
    rollout_agent = getattr(agent, "rollout_agent", None)
    rollout_policy = getattr(agent, "rollout_policy", None)

    if hasattr(agent, "rollout_policy_name"):
        rollout_name = agent.rollout_policy_name()
    elif rollout_agent is not None:
        rollout_name = rollout_agent.__class__.__name__
    elif rollout_policy is not None:
        rollout_name = str(rollout_policy)
    else:
        rollout_name = ""

    return {
        f"{prefix}_simulations": getattr(agent, "simulations", ""),
        f"{prefix}_selection_policy": (
            selection_policy.__class__.__name__
            if selection_policy is not None
            else ""
        ),
        f"{prefix}_exploration_constant": (
            getattr(selection_policy, "exploration_constant", "")
            if selection_policy is not None
            else ""
        ),
        f"{prefix}_heuristic_weight": (
            getattr(selection_policy, "heuristic_weight", "")
            if selection_policy is not None
            else ""
        ),
        f"{prefix}_thompson_prior_alpha": getattr(
            agent,
            "thompson_prior_alpha",
            "",
        ),
        f"{prefix}_thompson_prior_beta": getattr(
            agent,
            "thompson_prior_beta",
            "",
        ),
        f"{prefix}_rollout_policy": rollout_name,
        f"{prefix}_use_tactical_guard": getattr(
            agent,
            "use_tactical_guard",
            "",
        ),
        f"{prefix}_use_heuristic_expansion": getattr(
            agent,
            "use_heuristic_expansion",
            "",
        ),
        f"{prefix}_use_fork_detection": getattr(
            agent,
            "use_fork_detection",
            "",
        ),
    }


def completed_ok_seeds(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()

    completed: set[int] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(int(row["seed"]))

    return completed


def read_all_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    """Append one game result to the standard CSV."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    # Include a UTF-8 for Excel compatibility
    with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
        file.flush()

def side_for_seed(seed: int, side_mode: str) -> str:
    """Choose the tested side, alternating by seed when requested."""

    if side_mode == "alternate":
        return "X" if seed % 2 == 0 else "O"

    if side_mode in {"X", "O"}:
        return side_mode

    raise ValueError(
        "side_mode must be 'alternate', 'X', or 'O'."
    )


def winner_to_text(winner: object) -> str:
    if winner is None:
        return "DRAW"

    name = getattr(winner, "name", None)
    if name is not None:
        return name

    return str(winner)


def outcome_for_tested_agent(winner: object, tested_side: str) -> str:
    if winner is None:
        return "draw"

    tested_mark = Mark.X if tested_side == "X" else Mark.O

    if winner == tested_mark:
        return "win"

    return "loss"


def read_ok_rows(csv_path: Path) -> list[dict[str, str]]:
    """Return the latest successful row for each seed."""

    rows_by_seed: dict[int, dict[str, str]] = {}

    for row in read_all_rows(csv_path):
        if row.get("status") != "ok":
            continue

        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue

        rows_by_seed[seed] = row

    return [rows_by_seed[seed] for seed in sorted(rows_by_seed)]


def print_current_summary(csv_path: Path) -> None:
    rows = read_ok_rows(csv_path)
    outcomes = Counter(row["outcome_for_tested_agent"] for row in rows)
    total = sum(outcomes.values())

    if total == 0:
        print("No completed games yet.")
        return

    win_rate = outcomes["win"] / total
    draw_rate = outcomes["draw"] / total
    loss_rate = outcomes["loss"] / total

    runtimes = [float(row["runtime_seconds"]) for row in rows]
    avg_runtime = sum(runtimes) / len(runtimes)

    print(
        "Current summary: "
        f"completed={total}, "
        f"wins={outcomes['win']}, "
        f"draws={outcomes['draw']}, "
        f"losses={outcomes['loss']}, "
        f"win_rate={win_rate:.1%}, "
        f"draw_rate={draw_rate:.1%}, "
        f"loss_rate={loss_rate:.1%}, "
        f"avg_runtime={avg_runtime:.2f}s"
    )


def run_win_rate_experiment(
    *,
    experiment_name: str,
    comparison_file: str,
    board: BoardConfig,
    tested_agent_factory: AgentFactory,
    opponent_factory: AgentFactory,
    seeds: Iterable[int],
    output_csv: Path,
    side_mode: str = "alternate",
) -> None:
    """Run a resumable comparison and save each game to CSV."""

    seeds = list(seeds)
    completed = completed_ok_seeds(output_csv)

    runner = MatchRunner(
        rows=board.rows,
        cols=board.cols,
        win_length=board.win_length,
    )

    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Board: {board.rows}x{board.cols}, k={board.win_length}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} games total)")
    print(f"Side mode: {side_mode}")
    print(f"Output CSV: {output_csv}")
    print(f"Already completed seeds: {len(completed)}")
    print("=" * 80)

    for game_index, seed in enumerate(seeds, start=1):
        if seed in completed:
            print(f"[{game_index:03d}/{len(seeds):03d}] seed={seed} already done; skipping.")
            continue

        tested_side = side_for_seed(seed, side_mode)
        opponent_side = "O" if tested_side == "X" else "X"

        # creates new agents for new game
        tested_agent = tested_agent_factory()
        opponent = opponent_factory()

        if tested_side == "X":
            agent_x = tested_agent
            agent_o = opponent
        else:
            agent_x = opponent
            agent_o = tested_agent

        agent_x_name = display_agent_name(agent_x)
        agent_o_name = display_agent_name(agent_o)
        tested_agent_name = display_agent_name(tested_agent)
        opponent_name = display_agent_name(opponent)

        timestamp_start = datetime.now().isoformat(timespec="seconds")
        start_time = time.perf_counter()

        print(
            f"[{game_index:03d}/{len(seeds):03d}] "
            f"seed={seed}, tested_agent_as={tested_side} ..."
        )

        try:
            result = runner.play(
                agent_x=agent_x,
                agent_o=agent_o,
                seed=seed,
                record_trace=False,
            )

            runtime_seconds = time.perf_counter() - start_time
            timestamp_end = datetime.now().isoformat(timespec="seconds")
            moves_played = result.final_state.moves_played
            seconds_per_move = (
                runtime_seconds / moves_played
                if moves_played > 0
                else runtime_seconds
            )
            winner = result.winner
            outcome = outcome_for_tested_agent(
                winner=winner,
                tested_side=tested_side,
            )

            row: dict[str, object] = {
                "status": "ok",
                "experiment_name": experiment_name,
                "comparison_file": comparison_file,
                "game_index": game_index,
                "seed": seed,
                "board_rows": board.rows,
                "board_cols": board.cols,
                "win_length": board.win_length,
                "tested_agent_name": tested_agent_name,
                "opponent_name": opponent_name,
                "tested_agent_side": tested_side,
                "opponent_side": opponent_side,
                "agent_x_name": agent_x_name,
                "agent_o_name": agent_o_name,
                "winner": winner_to_text(winner),
                "outcome_for_tested_agent": outcome,
                "moves_played": moves_played,
                "runtime_seconds": f"{runtime_seconds:.6f}",
                "seconds_per_move": f"{seconds_per_move:.6f}",
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "error_message": "",
            }
            row.update(agent_config(tested_agent, "tested_agent"))
            row.update(agent_config(opponent, "opponent"))

            append_row(output_csv, row)
            completed.add(seed)

            print(
                f"    result={outcome}, winner={winner_to_text(winner)}, "
                f"moves={moves_played}, runtime={runtime_seconds:.2f}s"
            )
            print_current_summary(output_csv)

        except Exception as error:
            runtime_seconds = time.perf_counter() - start_time
            timestamp_end = datetime.now().isoformat(timespec="seconds")

            error_row: dict[str, object] = {
                "status": "error",
                "experiment_name": experiment_name,
                "comparison_file": comparison_file,
                "game_index": game_index,
                "seed": seed,
                "board_rows": board.rows,
                "board_cols": board.cols,
                "win_length": board.win_length,
                "tested_agent_name": tested_agent_name,
                "opponent_name": opponent_name,
                "tested_agent_side": tested_side,
                "opponent_side": opponent_side,
                "agent_x_name": agent_x_name,
                "agent_o_name": agent_o_name,
                "runtime_seconds": f"{runtime_seconds:.6f}",
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "error_message": repr(error),
            }
            error_row.update(agent_config(tested_agent, "tested_agent"))
            error_row.update(agent_config(opponent, "opponent"))

            append_row(output_csv, error_row)
            print(f"    ERROR after {runtime_seconds:.2f}s: {error!r}")
            print("    Previous completed games are already saved in the output files.")
            raise

    print("=" * 80)
    print(f"Finished experiment: {experiment_name}")
    print_current_summary(output_csv)
    print(f"Full CSV: {output_csv}")
    print("=" * 80)

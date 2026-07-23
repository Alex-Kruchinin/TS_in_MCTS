from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CURRENT_FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = CURRENT_FOLDER / "results"
PLOTS_FOLDER = CURRENT_FOLDER / "plots"
PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = RESULTS_FOLDER / "win_rate_summary.csv"
WIN_RATE_PNG = PLOTS_FOLDER / "win_rate_by_agent.png"
RUNTIME_PNG = PLOTS_FOLDER / "average_runtime_by_agent.png"


def read_result_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for csv_path in sorted(RESULTS_FOLDER.glob("*.csv")):
        if csv_path.name == SUMMARY_CSV.name:
            continue

        with csv_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("status") == "ok":
                    row["source_csv"] = csv_path.name
                    rows.append(row)

    return rows


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment_name"]].append(row)

    summary_rows: list[dict[str, object]] = []

    for experiment_name, experiment_rows in sorted(grouped.items()):
        outcomes = Counter(
            row["outcome_for_tested_agent"]
            for row in experiment_rows
        )
        total = len(experiment_rows)
        runtimes = [float(row["runtime_seconds"]) for row in experiment_rows]
        move_counts = [int(row["moves_played"]) for row in experiment_rows]

        tested_agent_name = experiment_rows[0]["tested_agent_name"]
        opponent_name = experiment_rows[0]["opponent_name"]

        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "tested_agent_name": tested_agent_name,
                "opponent_name": opponent_name,
                "games_completed": total,
                "wins": outcomes["win"],
                "draws": outcomes["draw"],
                "losses": outcomes["loss"],
                "win_rate": outcomes["win"] / total if total else 0.0,
                "draw_rate": outcomes["draw"] / total if total else 0.0,
                "loss_rate": outcomes["loss"] / total if total else 0.0,
                "average_runtime_seconds": (
                    sum(runtimes) / len(runtimes) if runtimes else 0.0
                ),
                "average_moves_played": (
                    sum(move_counts) / len(move_counts) if move_counts else 0.0
                ),
            }
        )

    return summary_rows


def save_summary(summary_rows: list[dict[str, object]]) -> None:
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        print("No completed result rows found.")
        return

    fieldnames = list(summary_rows[0].keys())
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary CSV: {SUMMARY_CSV}")


def short_label(experiment_name: str) -> str:
    label = experiment_name
    label = label.removeprefix("01_")
    label = label.removeprefix("02_")
    label = label.removeprefix("03_")
    label = label.removeprefix("04_")
    label = label.removeprefix("05_")
    label = label.removeprefix("06_")
    return label.replace("_", "\n")


def plot_win_rates(summary_rows: list[dict[str, object]]) -> None:
    labels = [short_label(str(row["experiment_name"])) for row in summary_rows]
    win_rates = [float(row["win_rate"]) * 100.0 for row in summary_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, win_rates)
    plt.ylabel("Win rate for tested agent (%)")
    plt.title("Win Rate by Agent / Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(WIN_RATE_PNG, dpi=200)
    plt.close()

    print(f"Saved win-rate graph: {WIN_RATE_PNG}")


def plot_average_runtimes(summary_rows: list[dict[str, object]]) -> None:
    labels = [short_label(str(row["experiment_name"])) for row in summary_rows]
    runtimes = [float(row["average_runtime_seconds"]) for row in summary_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, runtimes)
    plt.ylabel("Average runtime per game (seconds)")
    plt.title("Average Runtime by Agent / Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RUNTIME_PNG, dpi=200)
    plt.close()

    print(f"Saved runtime graph: {RUNTIME_PNG}")


if __name__ == "__main__":
    rows = read_result_rows()
    summary_rows = build_summary(rows)
    save_summary(summary_rows)

    if summary_rows:
        plot_win_rates(summary_rows)
        plot_average_runtimes(summary_rows)

        print("\nSummary:")
        for row in summary_rows:
            print(
                f"{row['experiment_name']}: "
                f"games={row['games_completed']}, "
                f"wins={row['wins']}, "
                f"draws={row['draws']}, "
                f"losses={row['losses']}, "
                f"win_rate={float(row['win_rate']):.1%}, "
                f"avg_runtime={float(row['average_runtime_seconds']):.2f}s"
            )

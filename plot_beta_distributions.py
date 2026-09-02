from __future__ import annotations

from math import gamma
from pathlib import Path

import matplotlib.pyplot as plt


DISTRIBUTIONS = (
    ("Child A: Beta(6,3)", 6.0, 3.0, "tab:blue"),
    ("Child B: Beta(3,2)", 3.0, 2.0, "tab:orange"),
    ("Child C: Beta(12,7)", 12.0, 7.0, "tab:green"),
)

SAMPLES = {
    "Child A": 0.62,
    "Child B": 0.82,
    "Child C": 0.59,
}

OUTPUT_FILE = Path(__file__).with_name("fig02_thompson_posteriors.png")


def beta_density(x: float, alpha: float, beta: float) -> float:

    if x <= 0.0 or x >= 1.0:
        return 0.0

    normalising_constant = gamma(alpha + beta) / (
        gamma(alpha) * gamma(beta)
    )
    return (
        normalising_constant
        * x ** (alpha - 1.0)
        * (1.0 - x) ** (beta - 1.0)
    )


def create_plot() -> None:

    x_values = [index / 1000.0 for index in range(1001)]

    figure, axis = plt.subplots(figsize=(10, 5))

    for label, alpha, beta, colour in DISTRIBUTIONS:
        y_values = [
            beta_density(x, alpha, beta)
            for x in x_values
        ]
        axis.plot(
            x_values,
            y_values,
            color=colour,
            label=label,
        )

    for sample in SAMPLES.values():
        axis.axvline(
            sample,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
        )

    axis.text(
        0.81,
        0.42,
        "largest sample \n select Child B",
        horizontalalignment="center",
        verticalalignment="center",
    )

    axis.set_title(
        "Illustrative Thompson samples from three child-node posteriors"
    )
    axis.set_xlabel("Possible success probability")
    axis.set_ylabel("Posterior density")
    axis.set_xlim(0.0, 1.0)
    axis.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(OUTPUT_FILE, dpi=100)

    print(f"Plot saved to: {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    create_plot()

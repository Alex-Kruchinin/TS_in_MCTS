# Win Rate by Agent Experiments

This folder contains long-running experiment scripts for dissertation win-rate data.
Each comparison is stored in a separate file so that a failure in one comparison does
not lose progress from the others.

## Comparison files

```text
01_baseline_uct_vs_tactical.py
02_enhanced_uct_vs_tactical.py
03_baseline_thompson_vs_tactical.py
04_enhanced_thompson_vs_tactical.py
05_enhanced_uct_vs_random.py
06_enhanced_thompson_vs_random.py
```

Each script uses seeds `0..99`, so it runs 100 games. With `SIDE_MODE = "alternate"`,
the tested agent plays 50 games as X and 50 games as O.

## Output files

Each script saves results after every game.

For example, running:

```text
01_baseline_uct_vs_tactical.py
```

creates:

```text
results/01_baseline_uct_vs_tactical.csv
results/01_baseline_uct_vs_tactical.tsv
results/01_baseline_uct_vs_tactical_compact.tsv
```

The `.csv` file is the detailed machine-readable file used by the plotting code and
resume logic.

The `.tsv` file is the same full data but tab-separated. It is easier to open in Excel
because each field appears in a separate spreadsheet column.

The `_compact.tsv` file contains only the most important columns for quick reading:
seed, side, winner, outcome, move count and runtime.

## Running correctly in PyCharm

Open one script, right click inside the editor, and choose normal Python Run.
Do not run these files as pytest tests.

A correct run starts with output similar to:

```text
Experiment: 01_baseline_uct_vs_tactical
Board: 7x7, k=4
Seeds: 0..99 (100 games total)
Output CSV: ...
Readable full TSV: ...
Readable compact TSV: ...
```

## Resume behaviour

The CSV file is used for resume. If the script is stopped after 60 games, rerunning it
will skip already completed seeds and continue with the remaining ones.

If you already have an old CSV but no TSV files, rerun the experiment script. It will
rebuild the TSV files from the existing CSV even if all seeds are already completed.

## Plotting

After one or more comparisons are complete, run:

```text
plot_win_rate_by_agent.py
```

It reads the detailed CSV files from `results/` and creates summary graphs in `plots/`.

## Important update: internal MCTS heuristics

Enhanced UCT and Enhanced Thompson now use `InternalHeuristicRollout` implemented
inside `MCTSAgent`. The MCTS agents do not call `TacticalAgent` during rollout or
heuristic expansion.

`TacticalAgent` is still used as the external opponent in the comparison scripts,
which is allowed because it is the benchmark agent being played against. It is not
embedded inside MCTS.

Because this changes the enhanced-agent implementation, rerun enhanced experiments
from a clean results folder before using the results in the dissertation.

# Tic-Tac-Toe MCTS project

This folder is self-contained and can sit inside a larger project as
`tic-tac-toe/`

## Project structure

```text
tic-tac-toe/
├── src/
│   ├── agents/
│   ├── games/
│   ├── matches/
│   ├── visualization/
│   └── experiments/
│       ├── random_benchmark/
│       └── win_rate_by_agent/
├── tests/
├── pytest.ini
└── README.md
```

## Pytest is required to run code

From the `tic-tac-toe` directory on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
```


## Main TacticalAgent benchmark

The main benchmark scripts for experiments are in:

```text
src/experiments/win_rate_by_agent/
```

The four individual agent comparisons against the `BenchmarkAgent`:

```text
baseline_uct_vs_tactical.py
enhanced_uct_vs_tactical.py
baseline_thompson_vs_tactical.py
enhanced_thompson_vs_tactical.py
```

To run all four in sequence in one file use:

```text
run_all_agents_vs_tactical.py
```

The board size, win line length, number of simulation,
seed, UCT parameters, Thompson sampling parameters, fork and
benchmark settings are kept in:

```text
benchmark_settings.py
```

## UCT and Thompson Sampling parameter test

The experiments on discovery of optimal Beta prior and Constant C.
Were conducted in:

```text
thompson_beta_sweep_vs_weak_tactical.py
uct_c_sweep_vs_weak_tactical.py
```

## Direct UCT vs Thompson comparisons

```text
baseline_thompson_vs_baseline_uct.py
enhanced_thompson_vs_enhanced_uct.py
```

The enhanced head-to-head file deliberately has its own contorl of: constant C,
progressive-bias weight, Beta prior, simulation
budget, seeds, and fork detection.
They can be changed for controlled experiments.

## Random benchmark

Experiments against `RandomAgent` are separated from the tactical benchmark.
They were used only for code running testing only

```text
src/experiments/random_benchmark/
├── enhanced_uct_vs_random.py
└── enhanced_thompson_vs_random.py
```

## Results

The results of all runs are stored in this folder after execution:

```text
src/experiments/win_rate_by_agent/results/
```

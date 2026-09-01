# Othello MCTS Project

This project contains an independent Othello implementation for comparing
UCT and Thompson Sampling in Monte Carlo Tree Search (MCTS).

The code supports scalable even-sized boards, legal move generation, captures
in all eight directions, forced passes, terminal-state detection, match
running, and reproducible experiments.

## Project structure

```text
othello/
├── src/
│   ├── game/          # Othello rules and board state
│   ├── agents/        # Random, tactical, UCT and Thompson agents
│   ├── matches/       # Match runner and result handling
│   ├── mcts/          # Classical and enhanced MCTS searches
│   └── evaluation/    # Parameter-study support
│
├── experiments/       # Repeatable experiments
├── scripts/           # Single-match demonstrations
├── results/           # CSV experiment results folder
├── tests/             
├── pyproject.toml     # Python package and test configuration
└── README.md
```

## Agents

The project includes:

- `RandomAgent`
- `WeakTacticalAgent`
- `TacticalAgent`
- `BenchmarkTacticalAgent`
- `MCTSAgent` using classical UCT
- `ThompsonMCTSAgent` using classical Thompson Sampling
- `EnhancedMCTSAgent` using enhanced UCT
- `EnhancedThompsonMCTSAgent` using enhanced Thompson Sampling

In the classical comparison, UCT and Thompson Sampling use the same expansion,
rollout, reward, backpropagation, and final-move rules.

The enhanced agents add Othello-specific guidance, including heuristic
expansion, heuristic rollouts, rollout depth limits, corner safety, and
opponent-response lookahead.

## Main CSV experiments

The main experiment files are in `experiments/`:

- `08_classical_thompson_vs_classical_uct.py` compares the classical agents.
- `09_classical_uct_c_sweep_vs_weak_tactical.py` tests UCT exploration constants.
- `10_classical_thompson_beta_prior_sweep_vs_weak_tactical.py` tests Thompson Beta priors.
- `11_enhanced_thompson_vs_enhanced_uct.py` compares the enhanced agents.
- `12_all_agents_vs_calibrated_benchmark.py` evaluates selected agents against the benchmark.
- `13_benchmark_calibration_pilot.py` checks the benchmark difficulty before final evaluation.

These files keep their main settings near the top. This includes the board
size, simulation budget, number of games, random seeds, and agent parameters.
The experiments use fixed seeds and alternate colours.

From the `othello` directory, run an experiment with:

```powershell
python experiments\08_classical_thompson_vs_classical_uct.py
```

For example, run the enhanced comparison with:

```powershell
python experiments\11_enhanced_thompson_vs_enhanced_uct.py
```

Experiment output is saved under `results/`. The principal outputs are CSV
files containing per-game results and summaries.

## Other experiment runners

The following files offer command-line options for smaller or more flexible
runs:

```powershell
python experiments\thompson_vs_uct.py --variant classical --games 20 --simulations 100
python experiments\thompson_vs_tactical.py --variant enhanced --games 20 --simulations 100
python experiments\run_enhanced_parameter_study.py --mode ablation --games 10
```


## Installation and tests

Python 3.11 or later is required. From the `othello` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```



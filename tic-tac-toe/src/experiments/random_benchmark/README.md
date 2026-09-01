# Random benchmark experiments

This folder contains the experiments in which MCTS agents are evaluated
against `RandomAgent`.

Files:

- `enhanced_uct_vs_random.py`
- `enhanced_thompson_vs_random.py`

Both scripts reuse the board, simulation budget, seed range, UCT/Thompson
parameters, and enhanced-agent fork switch from:

`src/experiments/win_rate_by_agent/benchmark_settings.py`

Random-benchmark output is written to this folder's own `results/`
directory. The historical `win_rate_by_agent/results/` directory is not
modified by this reorganisation.

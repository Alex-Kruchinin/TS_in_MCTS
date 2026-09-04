# Evaluating UCT and Thompson Sampling under Limited Simulation Budgets: The Effect of Domain-Informed Heuristics in Board Games

This repository contains the code and experiments for an MSc dissertation
investigating Thompson Sampling as an alternative tree-selection method in
Monte Carlo Tree Search (MCTS).

## UCT and Thompson Sampling

MCTS builds a search tree through four repeated stages: 
1) selection 
2) expansion 
3) simulation
4) backpropagation

This project focuse is to compare two alternative selection policies (UCT and Thompson Sampling), where the search decides which existing child node to visit next.
And how heuristically guided expansion and simulation affects the results and quality of the selection.

UCT selects the child with the largest confidence-bound score:

```text
mean reward + C * sqrt(ln(parent visits) / child visits)
```

The mean reward favours moves that have performed well, while the exploration
term favours moves that have received fewer visits. The exploration constant
`C` controls the balance between these two parts.

Thompson Sampling represents the estimated value of each child with a Beta
distribution. It draws one sample from every child's distribution and selects
the child with the largest sample:

```text
theta_i ~ Beta(alpha_i, beta_i)
```

Successful results (wins) increase `alpha`, while unsuccessful results (loses) increase
`beta`; a draw contributes equally to both. As more results are observed, the
distribution becomes more concentrated. This balances exploration and
exploitation through posterior uncertainty rather than a UCT bonus.

Both methods therefore solve the same selection problem in different ways:
UCT uses a confidence-bound formula, whereas Thompson Sampling uses
probability sampling.

## The code tests:

- classical Thompson MCTS against classical UCT-MCTS
- enhanced (heuristically guided) Thompson MCTS against enhanced (heuristically guided) UCT-MCTS
- all 4 agent variants against rule-based benchmark opponents
- the effect of the UCT exploration constant `C` and Thompson Beta prior
- performance under controlled simulation budgets
- win, loss, draw, score, runtime, and move-count results

The principal experiments use:

7 x 7 Tic-Tac-Toe with four marks required to win.
6 x 6 Othello. Both game implementations also support configurable board sizes.

## Fair comparison

The main experiments use equal simulation budgets, fixed random seeds, fresh
agent instances, and alternating starting sides or colours. Results are saved
per game so that overall and side-specific performance can be examined.

In the classical head-to-head experiments, expansion, simulation,
backpropagation, rewards, and final-action selection are kept the same. The
main experimental difference is whether tree selection uses UCT or Thompson
Sampling.

The enhanced comparisons give both agents the same game-specific search
features where possible. UCT progressive bias can be set to zero when the goal
is to compare the original selection-policy only.

## Repository structure

```text
COMP66060_Masters_Project/
│
├── tic-tac-toe/   # Scalable Tic-Tac-Toe implementation and experiments
├── othello/       # Scalable Othello implementation and experiments
└── README.md      # Project overview
```

Each game is self-contained and has its own README with more information.

## Tic-Tac-Toe experiments

The main Tic-Tac-Toe experiments are in:

```text
tic-tac-toe/src/experiments/win_rate_by_agent/
```

Important files include:

- `baseline_thompson_vs_baseline_uct.py` for the classical comparison;
- `enhanced_thompson_vs_enhanced_uct.py` for the enhanced comparison;
- `run_all_agents_vs_tactical.py` for evaluating all four MCTS agents against
  the tactical benchmark;
- `uct_c_sweep_vs_weak_tactical.py` for testing UCT exploration constants;
- `thompson_beta_sweep_vs_weak_tactical.py` for testing Thompson Beta priors;
- `benchmark_settings.py` for the shared board, agent, seed, and experiment
  settings.


## Othello experiments

The main Othello experiments are in:

```text
othello/experiments/
```

Important files include:

- `08_classical_thompson_vs_classical_uct.py` for the classical comparison;
- `09_classical_uct_c_sweep_vs_weak_tactical.py` for the UCT parameter sweep;
- `10_classical_thompson_beta_prior_sweep_vs_weak_tactical.py` for the
  Thompson-prior sweep;
- `11_enhanced_thompson_vs_enhanced_uct.py` for the enhanced comparison;
- `12_all_agents_vs_calibrated_benchmark.py` for comparison against the
  rule-based benchmark;
- `run_enhanced_parameter_study.py` for enhanced-agent ablation and parameter
  studies.


Python 3.11 or later is recommended.

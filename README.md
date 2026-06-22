Thompson Sampling in Monte Carlo Tree Search

This repository contains the implementation and experimental work for an MSc dissertation investigating the use of Thompson Sampling as a tree-selection strategy in Monte Carlo Tree Search.

Monte Carlo Tree Search, or MCTS, is a heuristic search algorithm commonly used for sequential decision-making problems, particularly in games with large search spaces. Standard MCTS typically uses the Upper Confidence Bound applied to Trees, or UCT, during the selection stage to balance exploration of less-visited actions with exploitation of actions that have previously produced strong results.

This project investigates whether Thompson Sampling can provide an effective alternative to UCT. Instead of selecting actions using a deterministic confidence-bound formula, Thompson Sampling models the uncertainty associated with each action and selects actions by sampling from probability distributions representing their estimated performance.

The two approaches will be evaluated across multiple games, board configurations, opponent types, and computational budgets.

Research Aim

The main aim of this project is to investigate the effectiveness of Thompson Sampling when used within Monte Carlo Tree Search.

The project will compare Thompson Sampling-based MCTS with standard UCT-based MCTS in terms of:

playing strength;
win, loss, and draw rates;
computational efficiency;
convergence behaviour;
robustness across different games;
performance under different simulation budgets;
performance against opponents of varying difficulty.
Research Questions

The project is guided by the following research questions:

How does Thompson Sampling-based MCTS compare with standard UCT-based MCTS in terms of playing performance?
How does the number of MCTS simulations affect the performance of each selection strategy?
Does Thompson Sampling perform consistently across games with different structures and branching factors?
How does each MCTS variant perform against opponents of different strengths?
How do board size and game complexity affect the relative performance of Thompson Sampling and UCT?
Games

The project currently focuses on the following games.

Scalable Tic-Tac-Toe

A configurable version of Tic-Tac-Toe will be implemented in which the board dimensions and winning-line requirements can be changed.

Possible configurations include:

standard (3 \times 3) Tic-Tac-Toe;
larger square boards;
configurable numbers of consecutive marks required to win.

Using multiple board sizes allows the complexity and branching factor of the game to be varied systematically.

Othello

Othello, also known as Reversi, will be used as a more strategically complex game with:

changing board control;
positional strategy;
variable legal moves;
pass actions;
a larger and more complex search space.

The use of both Tic-Tac-Toe and Othello allows the algorithms to be compared across games with different strategic properties.

Monte Carlo Tree Search

A standard MCTS iteration consists of four stages:

Selection
Starting from the root node, the algorithm repeatedly selects child nodes until it reaches a node that is not fully expanded or represents a terminal state.
Expansion
One or more previously unexplored actions are added to the search tree.
Simulation
A game is simulated from the selected state until a terminal state or predefined stopping condition is reached.
Backpropagation
The simulation result is propagated back through the visited nodes to update their statistics.

This project primarily investigates modifications to the selection stage.

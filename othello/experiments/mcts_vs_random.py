from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataclasses import dataclass

from agents import MCTSAgent, RandomAgent
from game import Disc, OthelloState
from matches import run_match


@dataclass
class Summary:
    mcts_wins: int = 0
    random_wins: int = 0
    draws: int = 0
    total_mcts_discs: int = 0
    total_random_discs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent Othello UCT-MCTS versus RandomAgent."
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--fixed-colour",
        choices=("black", "white"),
        default=None,
        help="Keep MCTS on one colour instead of alternating colours.",
    )
    return parser.parse_args()


def mcts_colour_for_game(game_index: int, fixed_colour: str | None) -> Disc:
    if fixed_colour == "black":
        return Disc.BLACK
    if fixed_colour == "white":
        return Disc.WHITE
    return Disc.BLACK if game_index % 2 == 0 else Disc.WHITE


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("Games must be greater than zero.")

    summary = Summary()

    for game_index in range(args.games):
        game_number = game_index + 1
        mcts_colour = mcts_colour_for_game(game_index, args.fixed_colour)

        mcts = MCTSAgent(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            seed=args.seed + game_index * 2,
            name="UCT-MCTS",
        )
        random_agent = RandomAgent(
            seed=args.seed + game_index * 2 + 1,
            name="Random",
        )

        if mcts_colour is Disc.BLACK:
            black_agent = mcts
            white_agent = random_agent
        else:
            black_agent = random_agent
            white_agent = mcts

        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(
                rows=args.rows,
                cols=args.cols,
            ),
        )

        if mcts_colour is Disc.BLACK:
            mcts_discs = result.black_count
            random_discs = result.white_count
        else:
            mcts_discs = result.white_count
            random_discs = result.black_count

        summary.total_mcts_discs += mcts_discs
        summary.total_random_discs += random_discs

        if result.winner is None:
            outcome = "DRAW"
            summary.draws += 1
        elif result.winner is mcts_colour:
            outcome = "MCTS WIN"
            summary.mcts_wins += 1
        else:
            outcome = "RANDOM WIN"
            summary.random_wins += 1

        print(
            f"Game {game_number:>2}: "
            f"MCTS={mcts_colour.name:<5} | "
            f"MCTS discs={mcts_discs:>2} | "
            f"Random discs={random_discs:>2} | "
            f"{outcome}"
        )

    print("\nSummary")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Simulations per MCTS move: {args.simulations}")
    print(f"Exploration constant: {args.exploration:.4f}")
    print(f"MCTS wins: {summary.mcts_wins}")
    print(f"Random wins: {summary.random_wins}")
    print(f"Draws: {summary.draws}")
    print(
        "Average discs: "
        f"MCTS={summary.total_mcts_discs / args.games:.2f}, "
        f"Random={summary.total_random_discs / args.games:.2f}"
    )


if __name__ == "__main__":
    main()

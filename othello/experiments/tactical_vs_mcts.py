from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import MCTSAgent, TacticalAgent
from game import Disc, OthelloState
from matches import run_match


@dataclass
class Summary:
    tactical_wins: int = 0
    mcts_wins: int = 0
    draws: int = 0
    tactical_discs: int = 0
    mcts_discs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run balanced Othello TacticalAgent versus UCT-MCTS games."
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--exact-endgame-empty", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--exploration", type=float, default=2 ** 0.5)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument(
        "--fixed-colour",
        choices=("black", "white"),
        default=None,
        help="Keep TacticalAgent on one colour instead of alternating.",
    )
    return parser.parse_args()


def tactical_colour_for_game(index: int, fixed: str | None) -> Disc:
    if fixed == "black":
        return Disc.BLACK
    if fixed == "white":
        return Disc.WHITE
    return Disc.BLACK if index % 2 == 0 else Disc.WHITE


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("Games must be greater than zero.")

    summary = Summary()

    for game_index in range(args.games):
        tactical_colour = tactical_colour_for_game(
            game_index,
            args.fixed_colour,
        )
        tactical = TacticalAgent(
            search_depth=args.depth,
            exact_endgame_empty=args.exact_endgame_empty,
            seed=args.seed + game_index * 2,
            name="Tactical",
        )
        mcts = MCTSAgent(
            simulations=args.simulations,
            exploration_constant=args.exploration,
            seed=args.seed + game_index * 2 + 1,
            name="UCT-MCTS",
        )

        if tactical_colour is Disc.BLACK:
            black_agent, white_agent = tactical, mcts
        else:
            black_agent, white_agent = mcts, tactical

        result = run_match(
            black_agent,
            white_agent,
            initial_state=OthelloState.new(args.rows, args.cols),
        )

        if tactical_colour is Disc.BLACK:
            tactical_discs = result.black_count
            mcts_discs = result.white_count
        else:
            tactical_discs = result.white_count
            mcts_discs = result.black_count

        summary.tactical_discs += tactical_discs
        summary.mcts_discs += mcts_discs

        if result.winner is None:
            outcome = "DRAW"
            summary.draws += 1
        elif result.winner is tactical_colour:
            outcome = "TACTICAL WIN"
            summary.tactical_wins += 1
        else:
            outcome = "MCTS WIN"
            summary.mcts_wins += 1

        print(
            f"Game {game_index + 1:>2}: "
            f"Tactical={tactical_colour.name:<5} | "
            f"Tactical discs={tactical_discs:>2} | "
            f"MCTS discs={mcts_discs:>2} | "
            f"{outcome}"
        )

    print("\nSummary")
    print(f"Board: {args.rows} x {args.cols}")
    print(f"Tactical search depth: {args.depth}")
    print(f"Exact endgame threshold: {args.exact_endgame_empty}")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"Tactical wins: {summary.tactical_wins}")
    print(f"MCTS wins: {summary.mcts_wins}")
    print(f"Draws: {summary.draws}")
    print(
        "Average discs: "
        f"Tactical={summary.tactical_discs / args.games:.2f}, "
        f"MCTS={summary.mcts_discs / args.games:.2f}"
    )


if __name__ == "__main__":
    main()

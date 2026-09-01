from __future__ import annotations

from dataclasses import dataclass

from game import Action, Disc, OthelloState


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Complete output of one Othello match."""

    initial_state: OthelloState
    final_state: OthelloState
    actions: tuple[Action, ...]
    winner: Disc | None
    black_count: int
    white_count: int
    pass_count: int

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def is_draw(self) -> bool:
        return self.winner is None

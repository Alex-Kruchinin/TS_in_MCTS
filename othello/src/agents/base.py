from __future__ import annotations

from typing import Protocol

from game import Action, Disc, OthelloState


class Agent(Protocol):
    """Interface implemented by every independent Othello agent."""

    @property
    def name(self) -> str:
        """Readable name used in traces and experiment results."""
        ...

    def choose_action(self, state: OthelloState) -> Action:
        """Choose one action from ``state.legal_actions()``."""
        ...

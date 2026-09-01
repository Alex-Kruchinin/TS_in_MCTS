from __future__ import annotations

import random

from game import Action, OthelloState


class RandomAgent:
    """Select uniformly from legal actions with a seeded generator"""

    def __init__(self, seed: int | None = None, name: str = "RandomAgent") -> None:
        self._rng = random.Random(seed)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choose_action(self, state: OthelloState) -> Action:
        legal_actions = state.legal_actions()
        if not legal_actions:
            raise ValueError("Cannot choose an action from a terminal state.")
        return self._rng.choice(legal_actions)

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias


class Disc(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    @property
    def opponent(self) -> "Disc":
        if self is Disc.BLACK:
            return Disc.WHITE
        if self is Disc.WHITE:
            return Disc.BLACK
        raise ValueError("EMPTY does not have an opponent.")

    @property
    def symbol(self) -> str:
        return {
            Disc.EMPTY: ".",
            Disc.BLACK: "B",
            Disc.WHITE: "W",
        }[self]


@dataclass(frozen=True, slots=True, order=True)
class Move:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class PassMove:
    def __str__(self) -> str:
        return "PASS"


PASS = PassMove()

Action: TypeAlias = Move | PassMove

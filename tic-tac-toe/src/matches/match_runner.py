from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Optional

from src.agents.base_agent import Agent
from src.games.tic_tac_toe import Mark, Move, TicTacToeState
from src.matches.match_trace import MatchTrace, TraceStep


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of a completed match, including moves, seed and optional trace."""

    final_state: TicTacToeState
    winner: Optional[Mark]
    move_history: tuple[Move, ...]
    seed: Optional[int]
    trace: Optional[MatchTrace] = None

    @property
    def number_of_moves(self) -> int:
        """Return the total number of moves played in the match."""

        return len(self.move_history)

    @property
    def is_draw(self) -> bool:
        """Return True when the completed match ended in a draw."""

        return self.final_state.is_draw()


@dataclass(frozen=True, slots=True)
class MatchRunner:
    """Run a complete Tic-Tac-Toe match between two agents."""

    rows: int = 3
    cols: int = 3
    win_length: int = 3

    def play(
        self,
        agent_x: Agent,
        agent_o: Agent,
        seed: Optional[int] = None,
        record_trace: bool = False,
    ) -> MatchResult:
        """Play one match from an empty board, with X moving first."""

        rng = Random(seed)

        initial_state = TicTacToeState.new(
            rows=self.rows,
            cols=self.cols,
            win_length=self.win_length,
        )

        state = initial_state
        move_history: list[Move] = []
        trace_steps: list[TraceStep] = []

        while not state.is_terminal():
            # Select the current player's agent.
            if state.player_to_move == Mark.X:
                current_agent = agent_x
            else:
                current_agent = agent_o

            state_before_move = state
            player = state.player_to_move
            agent_name = self._agent_name(current_agent)

            # Ask the agent for a move.
            move = current_agent.choose_move(state, rng)

            # Reject invalid agent output.
            if not state.is_legal_move(move):
                raise ValueError(
                    f"Agent for {state.player_to_move.name} returned "
                    f"illegal move ({move.row}, {move.col})."
                )

            move_history.append(move)
            state = state.apply_move(move)

            if record_trace:
                trace_steps.append(
                    TraceStep(
                        move_number=len(move_history),
                        player=player,
                        agent_name=agent_name,
                        move=move,
                        state_before_move=state_before_move,
                        state_after_move=state,
                    )
                )

        trace = None
        if record_trace:
            trace = MatchTrace(
                initial_state=initial_state,
                steps=tuple(trace_steps),
                final_state=state,
                winner=state.winner(),
                seed=seed,
                agent_x_name=self._agent_name(agent_x),
                agent_o_name=self._agent_name(agent_o),
            )

        return MatchResult(
            final_state=state,
            winner=state.winner(),
            move_history=tuple(move_history),
            seed=seed,
            trace=trace,
        )

    @staticmethod
    def _agent_name(agent: Agent) -> str:
        """Return a readable agent name for traces and debugging."""

        custom_text = str(agent)
        default_prefix = f"<{agent.__class__.__module__}."

        if not custom_text.startswith(default_prefix):
            return custom_text

        return agent.__class__.__name__

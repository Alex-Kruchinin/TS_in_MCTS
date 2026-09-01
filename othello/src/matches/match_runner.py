from __future__ import annotations

from collections.abc import Callable

from agents import Agent
from game import Action, Disc, OthelloState, PassMove
from matches.result import MatchResult

TraceCallback = Callable[[int, OthelloState, Action, str], None]


class MatchRunner:
    """Run a complete game between two Othello agents."""

    def __init__(
        self,
        black_agent: Agent,
        white_agent: Agent,
        *,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        self._agents = {
            Disc.BLACK: black_agent,
            Disc.WHITE: white_agent,
        }
        self._trace_callback = trace_callback

    def run(self, initial_state: OthelloState | None = None) -> MatchResult:
        state = initial_state if initial_state is not None else OthelloState.new()
        starting_state = state
        actions: list[Action] = []

        # Guard against accidental game loops.
        maximum_actions = state.rows * state.cols * 2

        while not state.is_terminal():
            if len(actions) >= maximum_actions:
                raise RuntimeError(
                    "Match exceeded its safety action limit; "
                    "the game or an agent may contain a loop."
                )

            player = state.current_player
            agent = self._agents[player]
            legal_actions = state.legal_actions()
            action = agent.choose_action(state)

            if action not in legal_actions:
                raise ValueError(
                    f"{agent.name} selected illegal action {action!r} "
                    f"for {player.name}. Legal actions: {legal_actions!r}."
                )

            if self._trace_callback is not None:
                self._trace_callback(
                    len(actions) + 1,
                    state,
                    action,
                    agent.name,
                )

            state = state.apply(action)
            actions.append(action)

        return MatchResult(
            initial_state=starting_state,
            final_state=state,
            actions=tuple(actions),
            winner=state.winner(),
            black_count=state.disc_count(Disc.BLACK),
            white_count=state.disc_count(Disc.WHITE),
            pass_count=sum(isinstance(action, PassMove) for action in actions),
        )


def run_match(
    black_agent: Agent,
    white_agent: Agent,
    *,
    initial_state: OthelloState | None = None,
    trace_callback: TraceCallback | None = None,
) -> MatchResult:
    """Run one complete Othello match."""

    return MatchRunner(
        black_agent,
        white_agent,
        trace_callback=trace_callback,
    ).run(initial_state)

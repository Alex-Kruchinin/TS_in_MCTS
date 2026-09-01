from __future__ import annotations

from random import Random

from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.games.tic_tac_toe import Mark, Move, TicTacToeState
from src.matches.match_runner import MatchRunner


class ScriptedAgent:
    """Test-only agent that returns moves from a predefined sequence."""

    def __init__(self, moves: list[Move]) -> None:
        self.moves = iter(moves)

    def choose_move(
        self,
        state: TicTacToeState,
        rng: Random,
    ) -> Move:
        return next(self.moves)


def test_match_result_has_no_trace_by_default() -> None:
    runner = MatchRunner()

    result = runner.play(
        agent_x=RandomAgent(),
        agent_o=RandomAgent(),
        seed=5,
    )

    assert result.trace is None


def test_match_runner_records_trace_when_requested() -> None:
    runner = MatchRunner()

    agent_x = ScriptedAgent(
        [Move(0, 0), Move(0, 1), Move(0, 2)]
    )
    agent_o = ScriptedAgent(
        [Move(1, 0), Move(1, 1)]
    )

    result = runner.play(
        agent_x=agent_x,
        agent_o=agent_o,
        seed=7,
        record_trace=True,
    )

    assert result.trace is not None
    assert result.trace.number_of_moves == result.number_of_moves
    assert result.trace.final_state == result.final_state
    assert result.trace.winner == Mark.X
    assert len(result.trace.steps) == 5

    first_step = result.trace.steps[0]

    assert first_step.move_number == 1
    assert first_step.player == Mark.X
    assert first_step.move == Move(0, 0)
    assert first_step.state_before_move.moves_played == 0
    assert first_step.state_after_move.moves_played == 1


def test_trace_text_contains_moves_boards_and_result() -> None:
    runner = MatchRunner()

    agent_x = ScriptedAgent(
        [Move(0, 0), Move(0, 1), Move(0, 2)]
    )
    agent_o = ScriptedAgent(
        [Move(1, 0), Move(1, 1)]
    )

    result = runner.play(
        agent_x=agent_x,
        agent_o=agent_o,
        seed=7,
        record_trace=True,
    )

    assert result.trace is not None

    text = result.trace.to_text()

    assert "Single Tic-Tac-Toe match trace" in text
    assert "Board: 3x3" in text
    assert "Initial board" in text
    assert "Move 1" in text
    assert "Move 5" in text
    assert "row 0, col 2" in text
    assert "Winner: X" in text
    assert "Moves played: 5" in text
    assert "X X X" in text


def test_trace_agent_name_includes_mcts_simulation_budget() -> None:
    runner = MatchRunner()

    result = runner.play(
        agent_x=MCTSAgent(simulations=20),
        agent_o=RandomAgent(),
        seed=9,
        record_trace=True,
    )

    assert result.trace is not None
    assert result.trace.agent_x_name == "MCTSAgent(simulations=20)"
    assert result.trace.agent_o_name == "RandomAgent"

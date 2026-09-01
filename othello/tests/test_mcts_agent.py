from agents import MCTSAgent, RandomAgent
from game import Disc, OthelloState, PASS
from matches import run_match


def test_mcts_agent_returns_legal_action() -> None:
    state = OthelloState.new(rows=4, cols=4)
    agent = MCTSAgent(simulations=20, seed=10)

    action = agent.choose_action(state)

    assert action in state.legal_actions()
    assert agent.last_root is not None
    assert agent.last_root.visits == 20


def test_seeded_mcts_is_reproducible() -> None:
    state = OthelloState.new(rows=4, cols=4)

    first = MCTSAgent(simulations=30, seed=99).choose_action(state)
    second = MCTSAgent(simulations=30, seed=99).choose_action(state)

    assert first == second


def test_mcts_handles_forced_pass() -> None:
    state = OthelloState.from_strings(
        [
            "WWWW",
            "WWWW",
            "WWBW",
            "WW.W",
        ],
        current_player=Disc.BLACK,
    )

    action = MCTSAgent(simulations=10, seed=3).choose_action(state)

    assert action == PASS


def test_mcts_can_complete_match_against_random_agent() -> None:
    result = run_match(
        MCTSAgent(simulations=40, seed=1),
        RandomAgent(seed=2),
        initial_state=OthelloState.new(rows=4, cols=4),
    )

    assert result.final_state.is_terminal()
    assert result.action_count > 0
    assert result.black_count + result.white_count <= 16

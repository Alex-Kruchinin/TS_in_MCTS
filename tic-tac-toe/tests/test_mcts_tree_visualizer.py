from random import Random

from src.agents.mcts_agent import MCTSAgent
from src.games.tic_tac_toe import TicTacToeState
from src.visualization.mcts_tree_visualizer import MCTSTreeVisualizer


def test_visualizer_exports_dot_file(tmp_path) -> None:
    state = TicTacToeState.new()
    agent = MCTSAgent.baseline_uct(simulations=10)
    root = agent.search(state, Random(7))

    visualizer = MCTSTreeVisualizer(
        max_depth=1,
        top_k_children=3,
        include_boards=True,
    )

    output_path = tmp_path / "tree.dot"
    visualizer.export_dot(root, output_path)

    text = output_path.read_text(encoding="utf-8")

    assert "digraph MCTS" in text
    assert "ROOT" in text
    assert "visits=10" in text
    assert "alpha=" in text
    assert "beta=" in text


def test_visualizer_rejects_invalid_depth() -> None:
    try:
        MCTSTreeVisualizer(max_depth=-1)
    except ValueError as error:
        assert "max_depth" in str(error)
    else:
        raise AssertionError("Expected ValueError for negative max_depth.")

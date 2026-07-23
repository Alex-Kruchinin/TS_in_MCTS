from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from src.agents.mcts_node import MCTSNode
from src.games.tic_tac_toe import Mark, Move, TicTacToeState


class MCTSTreeVisualizer:
    """
    Export a readable snapshot of an MCTS tree.

    The full tree can become very large, so this visualiser deliberately
    limits both:
        - maximum depth shown;
        - number of children shown per node.

    The output is a Graphviz DOT file. If the Graphviz `dot` executable is
    installed, the same function can also render PNG, SVG or PDF files.
    """

    def __init__(
        self,
        max_depth: int = 2,
        top_k_children: int = 4,
        include_boards: bool = True,
        sort_children_by: str = "visits",
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        if top_k_children <= 0:
            raise ValueError("top_k_children must be positive.")

        valid_sort_keys = {"visits", "mean_value", "posterior_mean"}
        if sort_children_by not in valid_sort_keys:
            raise ValueError(
                "sort_children_by must be one of: "
                f"{sorted(valid_sort_keys)}"
            )

        self.max_depth = max_depth
        self.top_k_children = top_k_children
        self.include_boards = include_boards
        self.sort_children_by = sort_children_by

    def export_dot(
        self,
        root: MCTSNode,
        output_path: str | Path,
    ) -> Path:
        """Write the tree snapshot to a Graphviz DOT file."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "digraph MCTS {",
            "  graph [rankdir=TB, bgcolor=white];",
            "  node [shape=box, style=rounded, fontname=\"Consolas\", fontsize=10];",
            "  edge [fontname=\"Consolas\", fontsize=9];",
        ]

        node_ids: dict[int, str] = {}
        counter = 0

        def get_node_id(node: MCTSNode) -> str:
            nonlocal counter
            key = id(node)
            if key not in node_ids:
                node_ids[key] = f"n{counter}"
                counter += 1
            return node_ids[key]

        def walk(node: MCTSNode, depth: int) -> None:
            node_id = get_node_id(node)
            label = self._node_label(node=node, depth=depth, is_root=node is root)
            lines.append(
                f'  {node_id} [label="{self._escape_dot_label(label)}"];'
            )

            if depth >= self.max_depth:
                hidden_count = len(node.children)
                if hidden_count:
                    hidden_id = f"{node_id}_hidden"
                    lines.append(
                        f'  {hidden_id} [label="... {hidden_count} child nodes hidden", '
                        'shape=note, fontname="Consolas", fontsize=9];'
                    )
                    lines.append(f"  {node_id} -> {hidden_id} [style=dashed];")
                return

            visible_children = self._top_children(node)

            for child in visible_children:
                child_id = get_node_id(child)
                edge_label = self._edge_label(child.move)
                lines.append(
                    f'  {node_id} -> {child_id} '
                    f'[label="{self._escape_dot_label(edge_label)}"];'
                )
                walk(child, depth + 1)

            hidden_count = len(node.children) - len(visible_children)
            if hidden_count > 0:
                hidden_id = f"{node_id}_more"
                lines.append(
                    f'  {hidden_id} [label="... {hidden_count} more children", '
                    'shape=note, fontname="Consolas", fontsize=9];'
                )
                lines.append(f"  {node_id} -> {hidden_id} [style=dashed];")

        walk(root, depth=0)
        lines.append("}")

        output.write_text("\n".join(lines), encoding="utf-8")
        return output

    def export_graph(
        self,
        root: MCTSNode,
        output_path: str | Path,
        image_format: str = "png",
    ) -> tuple[Path, Path | None]:
        """
        Export DOT and, when Graphviz is available, render an image.

        Returns:
            (dot_path, image_path_or_none)
        """

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        dot_path = output.with_suffix(".dot")
        image_path = output.with_suffix(f".{image_format}")

        self.export_dot(root=root, output_path=dot_path)

        dot_executable = shutil.which("dot")
        if dot_executable is None:
            return dot_path, None

        subprocess.run(
            [
                dot_executable,
                f"-T{image_format}",
                str(dot_path),
                "-o",
                str(image_path),
            ],
            check=True,
        )

        return dot_path, image_path

    def _top_children(self, node: MCTSNode) -> list[MCTSNode]:
        """Return the most important children according to the sort key."""

        def sort_key(child: MCTSNode) -> tuple[float, float, float]:
            if self.sort_children_by == "mean_value":
                primary = child.mean_value
            elif self.sort_children_by == "posterior_mean":
                primary = child.posterior_mean
            else:
                primary = child.visits

            return (
                primary,
                child.visits,
                child.mean_value,
            )

        return sorted(
            node.children.values(),
            key=sort_key,
            reverse=True,
        )[: self.top_k_children]

    def _node_label(
        self,
        node: MCTSNode,
        depth: int,
        is_root: bool,
    ) -> str:
        if is_root:
            title = "ROOT"
        else:
            title = f"Move {self._format_move(node.move)}"

        player = node.player_just_moved.name

        lines = [
            title,
            f"depth={depth}",
            f"player_just_moved={player}",
            f"visits={node.visits}",
            f"mean={node.mean_value:.3f}",
            f"alpha={node.alpha:.2f}",
            f"beta={node.beta:.2f}",
            f"posterior_mean={node.posterior_mean:.3f}",
            f"heuristic={node.heuristic_value:.3f}",
        ]

        if self.include_boards:
            lines.extend(["", self._compact_board(node.state)])

        return "\n".join(lines)

    @staticmethod
    def _edge_label(move: Move | None) -> str:
        if move is None:
            return "root"
        return MCTSTreeVisualizer._format_move(move)

    @staticmethod
    def _format_move(move: Move | None) -> str:
        if move is None:
            return "None"
        return f"({move.row},{move.col})"

    @staticmethod
    def _compact_board(state: TicTacToeState) -> str:
        symbols = {
            Mark.EMPTY: ".",
            Mark.X: "X",
            Mark.O: "O",
        }

        rows = []
        for row in range(state.rows):
            values = []
            for col in range(state.cols):
                values.append(symbols[state.cell_at(Move(row, col))])
            rows.append(" ".join(values))

        return "\n".join(rows)

    @staticmethod
    def _escape_dot_label(label: str) -> str:
        return (
            label
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )


def export_mcts_tree(
    root: MCTSNode,
    output_path: str | Path,
    max_depth: int = 2,
    top_k_children: int = 4,
    include_boards: bool = True,
    sort_children_by: str = "visits",
    image_format: str = "png",
) -> tuple[Path, Path | None]:
    """
    Convenience function for exporting an MCTS tree snapshot.

    Returns:
        (dot_path, image_path_or_none)
    """

    visualizer = MCTSTreeVisualizer(
        max_depth=max_depth,
        top_k_children=top_k_children,
        include_boards=include_boards,
        sort_children_by=sort_children_by,
    )

    return visualizer.export_graph(
        root=root,
        output_path=output_path,
        image_format=image_format,
    )

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Iterable, Literal

from agents import EnhancedMCTSAgent, MCTSAgent, RandomAgent, TacticalAgent
from game import Disc, OthelloState
from matches import run_match

OpponentName = Literal["classical", "tactical", "random"]
AgentKind = Literal["enhanced", "classical"]


@dataclass(frozen=True, slots=True)
class AgentConfiguration:
    name: str
    agent_kind: AgentKind = "enhanced"
    simulations: int = 100
    exploration_constant: float = 2 ** 0.5
    progressive_bias_weight: float = 0.75
    rollout_epsilon: float = 0.15
    expansion_epsilon: float = 0.05
    rollout_depth_limit: int = 16
    guided_expansion: bool = True
    heuristic_rollouts: bool = True
    root_safety: bool = True
    root_lookahead: bool = True
    heuristic_cutoff: bool = True

    def build_agent(self, *, seed: int):
        if self.agent_kind == "classical":
            return MCTSAgent(
                simulations=self.simulations,
                exploration_constant=self.exploration_constant,
                seed=seed,
                name=self.name,
            )

        return EnhancedMCTSAgent(
            simulations=self.simulations,
            exploration_constant=self.exploration_constant,
            progressive_bias_weight=self.progressive_bias_weight,
            rollout_epsilon=self.rollout_epsilon,
            expansion_epsilon=self.expansion_epsilon,
            rollout_depth_limit=self.rollout_depth_limit,
            guided_expansion=self.guided_expansion,
            heuristic_rollouts=self.heuristic_rollouts,
            root_safety=self.root_safety,
            root_lookahead=self.root_lookahead,
            heuristic_cutoff=self.heuristic_cutoff,
            seed=seed,
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class StudyOutputs:
    game_results_csv: Path
    summary_csv: Path
    summary_markdown: Path
    metadata_json: Path


@dataclass(slots=True)
class _Aggregate:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    agent_discs: int = 0
    opponent_discs: int = 0
    total_actions: int = 0
    total_passes: int = 0
    total_seconds: float = 0.0

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def average_disc_difference(self) -> float:
        return (self.agent_discs - self.opponent_discs) / self.games


GAME_FIELDS = [
    "configuration",
    "agent_kind",
    "opponent",
    "game_number",
    "agent_colour",
    "outcome",
    "winner",
    "agent_discs",
    "opponent_discs",
    "disc_difference",
    "actions",
    "passes",
    "duration_seconds",
    "agent_seed",
    "opponent_seed",
    "simulations",
    "exploration_constant",
    "progressive_bias_weight",
    "rollout_epsilon",
    "expansion_epsilon",
    "rollout_depth_limit",
    "guided_expansion",
    "heuristic_rollouts",
    "root_safety",
    "root_lookahead",
    "heuristic_cutoff",
]

SUMMARY_FIELDS = [
    "rank",
    "configuration",
    "agent_kind",
    "opponent",
    "games",
    "wins",
    "losses",
    "draws",
    "score_rate",
    "win_rate",
    "average_agent_discs",
    "average_opponent_discs",
    "average_disc_difference",
    "average_actions",
    "average_passes",
    "average_game_seconds",
    "simulations",
    "exploration_constant",
    "progressive_bias_weight",
    "rollout_epsilon",
    "expansion_epsilon",
    "rollout_depth_limit",
    "guided_expansion",
    "heuristic_rollouts",
    "root_safety",
    "root_lookahead",
    "heuristic_cutoff",
]


def ablation_configurations(
    *,
    simulations: int = 100,
    exploration_constant: float = 2 ** 0.5,
    progressive_bias_weight: float = 0.75,
    rollout_epsilon: float = 0.15,
    expansion_epsilon: float = 0.05,
    rollout_depth_limit: int = 16,
) -> tuple[AgentConfiguration, ...]:
    full = AgentConfiguration(
        name="full_enhanced",
        simulations=simulations,
        exploration_constant=exploration_constant,
        progressive_bias_weight=progressive_bias_weight,
        rollout_epsilon=rollout_epsilon,
        expansion_epsilon=expansion_epsilon,
        rollout_depth_limit=rollout_depth_limit,
    )
    return (
        full,
        replace(full, name="no_progressive_bias", progressive_bias_weight=0.0),
        replace(full, name="random_expansion", guided_expansion=False),
        replace(full, name="random_rollouts", heuristic_rollouts=False),
        replace(full, name="no_root_safety", root_safety=False),
        replace(full, name="no_root_lookahead", root_lookahead=False),
        replace(full, name="terminal_rollouts", heuristic_cutoff=False),
        AgentConfiguration(
            name="classical_uct_baseline",
            agent_kind="classical",
            simulations=simulations,
            exploration_constant=exploration_constant,
        ),
    )


def one_factor_configurations(
    *,
    base: AgentConfiguration,
    simulation_values: Iterable[int],
    progressive_bias_values: Iterable[float],
    rollout_epsilon_values: Iterable[float],
    rollout_depth_values: Iterable[int],
) -> tuple[AgentConfiguration, ...]:
    candidates: list[AgentConfiguration] = [base]
    candidates.extend(
        replace(base, name=f"simulations_{value}", simulations=value)
        for value in simulation_values
    )
    candidates.extend(
        replace(
            base,
            name=f"progressive_bias_{value:g}",
            progressive_bias_weight=value,
        )
        for value in progressive_bias_values
    )
    candidates.extend(
        replace(
            base,
            name=f"rollout_epsilon_{value:g}",
            rollout_epsilon=value,
        )
        for value in rollout_epsilon_values
    )
    candidates.extend(
        replace(
            base,
            name=f"rollout_depth_{value}",
            rollout_depth_limit=value,
        )
        for value in rollout_depth_values
    )

    unique: dict[tuple[object, ...], AgentConfiguration] = {}
    for config in candidates:
        identity = (
            config.agent_kind,
            config.simulations,
            config.exploration_constant,
            config.progressive_bias_weight,
            config.rollout_epsilon,
            config.expansion_epsilon,
            config.rollout_depth_limit,
            config.guided_expansion,
            config.heuristic_rollouts,
            config.root_safety,
            config.root_lookahead,
            config.heuristic_cutoff,
        )
        unique.setdefault(identity, config)
    return tuple(unique.values())


def _opponent_agent(
    opponent: OpponentName,
    *,
    simulations: int,
    exploration_constant: float,
    tactical_depth: int,
    exact_endgame_empty: int,
    seed: int,
):
    if opponent == "classical":
        return MCTSAgent(
            simulations=simulations,
            exploration_constant=exploration_constant,
            seed=seed,
            name="Classical UCT opponent",
        )
    if opponent == "tactical":
        return TacticalAgent(
            search_depth=tactical_depth,
            exact_endgame_empty=exact_endgame_empty,
            seed=seed,
            name="Tactical opponent",
        )
    if opponent == "random":
        return RandomAgent(seed=seed, name="Random opponent")
    raise ValueError(f"Unsupported opponent: {opponent}")


def _colour_for_game(game_index: int) -> Disc:
    return Disc.BLACK if game_index % 2 == 0 else Disc.WHITE


def _config_values(config: AgentConfiguration) -> dict[str, object]:
    values = asdict(config)
    values.pop("name")
    values.pop("agent_kind")
    return values


def run_parameter_study(
    configurations: Iterable[AgentConfiguration],
    *,
    opponent: OpponentName,
    games_per_configuration: int,
    rows: int,
    cols: int,
    seed: int,
    output_directory: Path,
    opponent_simulations: int | None = None,
    opponent_exploration_constant: float = 2 ** 0.5,
    tactical_depth: int = 2,
    exact_endgame_empty: int = 8,
    verbose: bool = False,
) -> StudyOutputs:
    configs = tuple(configurations)
    if not configs:
        raise ValueError("At least one configuration is required.")
    if games_per_configuration <= 0:
        raise ValueError("Games per configuration must be greater than zero.")
    if games_per_configuration % 2 != 0:
        raise ValueError(
            "Games per configuration must be even for balanced colours."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    games_path = output_directory / "game_results.csv"
    summary_path = output_directory / "summary.csv"
    markdown_path = output_directory / "summary.md"
    metadata_path = output_directory / "metadata.json"

    aggregates: dict[str, _Aggregate] = {
        config.name: _Aggregate() for config in configs
    }

    with games_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAME_FIELDS)
        writer.writeheader()

        for config_index, config in enumerate(configs):
            aggregate = aggregates[config.name]
            for game_index in range(games_per_configuration):
                agent_colour = _colour_for_game(game_index)
                # Reuse seed pairs across configurations for fairer comparison.
                agent_seed = seed + game_index * 2
                opponent_seed = seed + game_index * 2 + 1
                agent = config.build_agent(seed=agent_seed)
                opponent_budget = (
                    opponent_simulations
                    if opponent_simulations is not None
                    else config.simulations
                )
                opponent_agent = _opponent_agent(
                    opponent,
                    simulations=opponent_budget,
                    exploration_constant=opponent_exploration_constant,
                    tactical_depth=tactical_depth,
                    exact_endgame_empty=exact_endgame_empty,
                    seed=opponent_seed,
                )

                if agent_colour is Disc.BLACK:
                    black_agent, white_agent = agent, opponent_agent
                else:
                    black_agent, white_agent = opponent_agent, agent

                start = time.perf_counter()
                result = run_match(
                    black_agent,
                    white_agent,
                    initial_state=OthelloState.new(rows, cols),
                )
                duration = time.perf_counter() - start

                if agent_colour is Disc.BLACK:
                    agent_discs = result.black_count
                    opponent_discs = result.white_count
                else:
                    agent_discs = result.white_count
                    opponent_discs = result.black_count

                if result.winner is None:
                    outcome = "draw"
                    aggregate.draws += 1
                elif result.winner is agent_colour:
                    outcome = "win"
                    aggregate.wins += 1
                else:
                    outcome = "loss"
                    aggregate.losses += 1

                aggregate.games += 1
                aggregate.agent_discs += agent_discs
                aggregate.opponent_discs += opponent_discs
                aggregate.total_actions += result.action_count
                aggregate.total_passes += result.pass_count
                aggregate.total_seconds += duration

                row = {
                    "configuration": config.name,
                    "agent_kind": config.agent_kind,
                    "opponent": opponent,
                    "game_number": game_index + 1,
                    "agent_colour": agent_colour.name,
                    "outcome": outcome,
                    "winner": "DRAW" if result.winner is None else result.winner.name,
                    "agent_discs": agent_discs,
                    "opponent_discs": opponent_discs,
                    "disc_difference": agent_discs - opponent_discs,
                    "actions": result.action_count,
                    "passes": result.pass_count,
                    "duration_seconds": f"{duration:.6f}",
                    "agent_seed": agent_seed,
                    "opponent_seed": opponent_seed,
                    **_config_values(config),
                }
                writer.writerow(row)

                if verbose:
                    print(
                        f"{config.name:<24} game {game_index + 1:>2} "
                        f"as {agent_colour.name:<5}: {outcome.upper():<4} "
                        f"({agent_discs}-{opponent_discs})"
                    )

            print(
                f"Completed {config_index + 1}/{len(configs)}: "
                f"{config.name} | score={aggregate.score_rate:.3f} | "
                f"disc diff={aggregate.average_disc_difference:+.2f}"
            )

    ranked = sorted(
        configs,
        key=lambda item: (
            aggregates[item.name].score_rate,
            aggregates[item.name].average_disc_difference,
        ),
        reverse=True,
    )

    summary_rows: list[dict[str, object]] = []
    for rank, config in enumerate(ranked, start=1):
        aggregate = aggregates[config.name]
        summary_rows.append(
            {
                "rank": rank,
                "configuration": config.name,
                "agent_kind": config.agent_kind,
                "opponent": opponent,
                "games": aggregate.games,
                "wins": aggregate.wins,
                "losses": aggregate.losses,
                "draws": aggregate.draws,
                "score_rate": f"{aggregate.score_rate:.6f}",
                "win_rate": f"{aggregate.win_rate:.6f}",
                "average_agent_discs": f"{aggregate.agent_discs / aggregate.games:.6f}",
                "average_opponent_discs": f"{aggregate.opponent_discs / aggregate.games:.6f}",
                "average_disc_difference": f"{aggregate.average_disc_difference:.6f}",
                "average_actions": f"{aggregate.total_actions / aggregate.games:.6f}",
                "average_passes": f"{aggregate.total_passes / aggregate.games:.6f}",
                "average_game_seconds": f"{aggregate.total_seconds / aggregate.games:.6f}",
                **_config_values(config),
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Enhanced MCTS parameter study\n\n")
        handle.write(
            f"Board: {rows} x {cols}  \n"
            f"Opponent: {opponent}  \n"
            f"Games per configuration: {games_per_configuration}  \n"
            f"Base seed: {seed}\n\n"
        )
        handle.write(
            "| Rank | Configuration | W-L-D | Score | Avg disc difference | "
            "Avg seconds |\n"
        )
        handle.write("|---:|---|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['rank']} | {row['configuration']} | "
                f"{row['wins']}-{row['losses']}-{row['draws']} | "
                f"{float(row['score_rate']):.3f} | "
                f"{float(row['average_disc_difference']):+.2f} | "
                f"{float(row['average_game_seconds']):.3f} |\n"
            )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "board": {"rows": rows, "cols": cols},
        "opponent": opponent,
        "games_per_configuration": games_per_configuration,
        "base_seed": seed,
        "opponent_simulations": opponent_simulations,
        "opponent_exploration_constant": opponent_exploration_constant,
        "tactical_depth": tactical_depth,
        "exact_endgame_empty": exact_endgame_empty,
        "configurations": [asdict(config) for config in configs],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return StudyOutputs(
        game_results_csv=games_path,
        summary_csv=summary_path,
        summary_markdown=markdown_path,
        metadata_json=metadata_path,
    )

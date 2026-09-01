from pathlib import Path

from evaluation import (
    AgentConfiguration,
    ablation_configurations,
    one_factor_configurations,
    run_parameter_study,
)


def test_ablation_configurations_are_unique_and_include_baselines() -> None:
    configurations = ablation_configurations(simulations=5)
    names = [configuration.name for configuration in configurations]

    assert len(names) == len(set(names))
    assert "full_enhanced" in names
    assert "no_progressive_bias" in names
    assert "random_expansion" in names
    assert "random_rollouts" in names
    assert "no_root_safety" in names
    assert "no_root_lookahead" in names
    assert "terminal_rollouts" in names
    assert "classical_uct_baseline" in names


def test_one_factor_configurations_remove_duplicate_baseline_values() -> None:
    base = AgentConfiguration(name="full_enhanced", simulations=10)
    configurations = one_factor_configurations(
        base=base,
        simulation_values=(10, 20),
        progressive_bias_values=(0.75, 1.0),
        rollout_epsilon_values=(0.15, 0.3),
        rollout_depth_values=(16, 24),
    )

    identities = {
        (
            item.simulations,
            item.progressive_bias_weight,
            item.rollout_epsilon,
            item.rollout_depth_limit,
        )
        for item in configurations
    }
    assert len(configurations) == len(identities)


def test_small_parameter_study_writes_all_outputs(tmp_path: Path) -> None:
    configurations = (
        AgentConfiguration(name="full", simulations=2, rollout_depth_limit=4),
        AgentConfiguration(
            name="random_rollout",
            simulations=2,
            rollout_depth_limit=4,
            heuristic_rollouts=False,
        ),
    )

    outputs = run_parameter_study(
        configurations,
        opponent="random",
        games_per_configuration=2,
        rows=4,
        cols=4,
        seed=123,
        output_directory=tmp_path,
    )

    assert outputs.game_results_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.summary_markdown.exists()
    assert outputs.metadata_json.exists()

    game_text = outputs.game_results_csv.read_text(encoding="utf-8")
    summary_text = outputs.summary_csv.read_text(encoding="utf-8")
    assert "full" in game_text
    assert "random_rollout" in game_text
    assert "rank,configuration" in summary_text

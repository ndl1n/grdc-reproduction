from __future__ import annotations

from pathlib import Path

import numpy as np

from grdc.config import load_config, config_from_dict
from grdc.envs import make_env


def test_all_configs_load() -> None:
    for path in Path("configs").rglob("*.json"):
        config = load_config(path)
        assert config.training.total_steps > 0


def test_serialized_legacy_config_keeps_original_attention_behavior():
    legacy = config_from_dict({"model": {"hidden_dim": 128}})
    assert not legacy.model.stable_role_attention
    assert not legacy.model.include_previous_action
    revised = config_from_dict({"model": {"paper_equations": True}})
    assert revised.model.stable_role_attention


def test_revised_baseline_has_same_environment_history_and_learning_settings():
    grdc = load_config("configs/revised/cooperative_navigation.json")
    mappo = load_config("configs/revised/mappo_cooperative_navigation.json")
    assert grdc.environment == mappo.environment
    assert grdc.model.paper_equations and mappo.model.paper_equations
    assert grdc.model.include_previous_action and mappo.model.include_previous_action
    left, right = grdc.to_dict()["training"], mappo.to_dict()["training"]
    for name in ["role_entropy_coef", "prototype_decorrelation_coef", "enable_pruning"]:
        left.pop(name)
        right.pop(name)
    assert left == right
    assert not mappo.model.use_roles
    assert not mappo.model.use_local_graph
    assert not mappo.model.use_intra_role_communication


def test_cooperative_navigation_shapes_and_determinism() -> None:
    config = load_config("configs/cooperative_navigation.json")
    env = make_env(config.environment, seed=7)
    first = env.reset(seed=42)
    second = env.reset(seed=42)
    np.testing.assert_allclose(first.observations, second.observations)
    assert first.observations.shape == (env.n_agents, env.obs_dim)
    assert first.state.shape == (env.state_dim,)
    assert first.adjacency.shape == (env.n_agents, env.n_agents)
    assert not np.diag(first.adjacency).any()
    step = env.step(np.zeros(env.n_agents, dtype=np.int64))
    assert np.isfinite(step.reward)


def test_predator_prey_shapes() -> None:
    config = load_config("configs/predator_prey.json")
    env = make_env(config.environment, seed=3)
    result = env.reset()
    assert result.observations.shape == (env.n_agents, env.obs_dim)
    assert result.available_actions.all()


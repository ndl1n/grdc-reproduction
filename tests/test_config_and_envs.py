from __future__ import annotations

from pathlib import Path

import numpy as np

from grdc.config import load_config
from grdc.envs import make_env


def test_all_configs_load() -> None:
    for path in Path("configs").glob("*.json"):
        config = load_config(path)
        assert config.training.total_steps > 0


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


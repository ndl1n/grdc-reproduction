from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from grdc.config import EnvironmentConfig
from .base import MultiAgentEnv, ResetResult, StepResult


_RACE_UNITS = {
    "protoss": (["stalker", "zealot", "colossus"], [0.45, 0.45, 0.10], ["colossus"]),
    "terran": (["marine", "marauder", "medivac"], [0.45, 0.45, 0.10], ["medivac"]),
    "zerg": (["zergling", "hydralisk", "baneling"], [0.45, 0.45, 0.10], ["baneling"]),
}


class SMACv2Adapter(MultiAgentEnv):
    """Adapter for the official oxwhirl/smacv2 environment.

    Graph edges are computed from the raw allied-unit positions and each unit's
    official sight range. The edge is present when either endpoint sees the
    other, matching the visibility-constrained graph definition in GRDC.
    """

    def __init__(self, config: EnvironmentConfig, seed: int | None = None) -> None:
        try:
            from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
        except ImportError as exc:
            raise ImportError(
                "SMACv2 is not installed. Follow README.md: install StarCraft II, "
                "the SMAC_Maps folder, then `pip install git+https://github.com/oxwhirl/smacv2.git`."
            ) from exc

        if config.sc2_path:
            os.environ["SC2PATH"] = config.sc2_path
        self.config = config
        self.n_agents = config.n_agents
        self._seed_value = int(seed or 0)
        units, weights, exceptions = _RACE_UNITS[config.race]
        capability_config = {
            "n_units": config.n_agents,
            "n_enemies": config.n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": units,
                "exception_unit_types": exceptions,
                "weights": weights,
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": config.n_enemies,
                "map_x": 32,
                "map_y": 32,
            },
        }
        map_size = 10 if max(config.n_agents, config.n_enemies) <= 10 else 20
        self._env = StarCraftCapabilityEnvWrapper(
            capability_config=capability_config,
            map_name=f"{map_size}gen_{config.race}",
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            seed=self._seed_value,
        )
        info = self._env.get_env_info()
        self.n_agents = int(info["n_agents"])
        self.obs_dim = int(info["obs_shape"])
        self.state_dim = int(info["state_shape"])
        self.n_actions = int(info["n_actions"])
        self.episode_limit = int(info["episode_limit"])
        self._raw_env = self._locate_raw_env(self._env)

    @staticmethod
    def _locate_raw_env(env: Any) -> Any:
        current = env
        seen: set[int] = set()
        for _ in range(8):
            if id(current) in seen:
                break
            seen.add(id(current))
            if hasattr(current, "agents") and hasattr(current, "unit_sight_range"):
                return current
            next_env = None
            for name in ("env", "_env", "wrapped_env"):
                candidate = getattr(current, name, None)
                if candidate is not None and candidate is not current:
                    next_env = candidate
                    break
            if next_env is None:
                break
            current = next_env
        raise RuntimeError(
            "Could not access the official SMACv2 raw environment needed to build "
            "the visibility graph. The installed SMACv2 API may be incompatible."
        )

    def _adjacency(self) -> np.ndarray:
        adjacency = np.zeros((self.n_agents, self.n_agents), dtype=np.bool_)
        units = self._raw_env.agents
        for i in range(self.n_agents):
            unit_i = units[i]
            if unit_i.health <= 0:
                continue
            sight_i = float(self._raw_env.unit_sight_range(i))
            for j in range(i + 1, self.n_agents):
                unit_j = units[j]
                if unit_j.health <= 0:
                    continue
                sight_j = float(self._raw_env.unit_sight_range(j))
                distance = math.hypot(unit_i.pos.x - unit_j.pos.x, unit_i.pos.y - unit_j.pos.y)
                if distance <= sight_i or distance <= sight_j:
                    adjacency[i, j] = adjacency[j, i] = True
        return adjacency

    def _snapshot(self) -> ResetResult:
        return ResetResult(
            observations=np.asarray(self._env.get_obs(), dtype=np.float32),
            state=np.asarray(self._env.get_state(), dtype=np.float32),
            available_actions=np.asarray(self._env.get_avail_actions(), dtype=np.bool_),
            adjacency=self._adjacency(),
        )

    def reset(self, seed: int | None = None) -> ResetResult:
        if seed is not None and hasattr(self._env, "seed"):
            try:
                self._env.seed(seed)
            except TypeError:
                self._env.seed()
        self._env.reset()
        return self._snapshot()

    def step(self, actions: np.ndarray) -> StepResult:
        reward, terminated, info = self._env.step(np.asarray(actions, dtype=np.int64).tolist())
        snapshot = self._snapshot()
        truncated = bool(info.get("episode_limit", False))
        true_termination = bool(terminated and not truncated)
        return StepResult(
            **snapshot.__dict__, reward=float(reward), terminated=true_termination,
            truncated=truncated, info=dict(info),
        )

    def close(self) -> None:
        self._env.close()


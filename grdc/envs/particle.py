from __future__ import annotations

import numpy as np

from grdc.config import EnvironmentConfig
from .base import MultiAgentEnv, ResetResult, StepResult


_ACTION_DELTAS = np.asarray(
    [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
    dtype=np.float32,
)


class _ParticleBase(MultiAgentEnv):
    n_actions = 5

    def __init__(self, config: EnvironmentConfig, seed: int | None = None) -> None:
        self.config = config
        self.n_agents = config.n_agents
        self.episode_limit = config.episode_limit
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self.agent_pos = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros_like(self.agent_pos)

    def _seed(self, seed: int | None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _spawn(self, count: int) -> np.ndarray:
        margin = 0.05 * self.config.world_size
        return self._rng.uniform(
            -self.config.world_size + margin,
            self.config.world_size - margin,
            size=(count, 2),
        ).astype(np.float32)

    def _move_agents(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_agents,):
            raise ValueError(f"Expected actions shape {(self.n_agents,)}, got {actions.shape}")
        if np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError("Action outside the discrete range [0, 4]")
        desired = _ACTION_DELTAS[actions] * self.config.agent_speed
        self.agent_vel = 0.25 * self.agent_vel + 0.75 * desired
        self.agent_pos = np.clip(
            self.agent_pos + self.agent_vel,
            -self.config.world_size,
            self.config.world_size,
        ).astype(np.float32)

    def _adjacency(self) -> np.ndarray:
        distances = np.linalg.norm(
            self.agent_pos[:, None, :] - self.agent_pos[None, :, :], axis=-1
        )
        visible = distances <= self.config.vision_radius
        np.fill_diagonal(visible, False)
        return visible.astype(np.bool_)

    def _available_actions(self) -> np.ndarray:
        return np.ones((self.n_agents, self.n_actions), dtype=np.bool_)


class CooperativeNavigationEnv(_ParticleBase):
    """Local-observation Cooperative Navigation used for the paper-style CN run.

    The team reward is the negative sum of landmark-to-nearest-agent distances,
    plus collision penalties. Observations hide entities outside radius 0.3 by
    zeroing their features and exposing an explicit visibility bit.
    """

    def __init__(self, config: EnvironmentConfig, seed: int | None = None) -> None:
        super().__init__(config, seed)
        self.n_landmarks = config.n_targets
        self.landmark_pos = np.zeros((self.n_landmarks, 2), dtype=np.float32)
        self.obs_dim = 4 + 3 * self.n_landmarks + 5 * (self.n_agents - 1)
        self.state_dim = 4 * self.n_agents + 2 * self.n_landmarks

    def reset(self, seed: int | None = None) -> ResetResult:
        self._seed(seed)
        self._step = 0
        self.agent_pos = self._spawn(self.n_agents)
        self.agent_vel.fill(0.0)
        self.landmark_pos = self._spawn(self.n_landmarks)
        return self._result()

    def _observations(self) -> np.ndarray:
        observations: list[np.ndarray] = []
        for i in range(self.n_agents):
            features: list[float] = [*self.agent_vel[i], *self.agent_pos[i]]
            for landmark in self.landmark_pos:
                relative = landmark - self.agent_pos[i]
                visible = float(np.linalg.norm(relative) <= self.config.vision_radius)
                features.extend((relative * visible).tolist())
                features.append(visible)
            for j in range(self.n_agents):
                if i == j:
                    continue
                relative = self.agent_pos[j] - self.agent_pos[i]
                visible = float(np.linalg.norm(relative) <= self.config.vision_radius)
                features.extend((relative * visible).tolist())
                features.extend((self.agent_vel[j] * visible).tolist())
                features.append(visible)
            observations.append(np.asarray(features, dtype=np.float32))
        return np.stack(observations)

    def _state(self) -> np.ndarray:
        return np.concatenate(
            [self.agent_pos.reshape(-1), self.agent_vel.reshape(-1), self.landmark_pos.reshape(-1)]
        ).astype(np.float32)

    def _result(self) -> ResetResult:
        return ResetResult(
            observations=self._observations(),
            state=self._state(),
            available_actions=self._available_actions(),
            adjacency=self._adjacency(),
        )

    def step(self, actions: np.ndarray) -> StepResult:
        self._move_agents(actions)
        self._step += 1
        distances = np.linalg.norm(
            self.landmark_pos[:, None, :] - self.agent_pos[None, :, :], axis=-1
        )
        reward = -float(distances.min(axis=1).sum())
        pairwise = np.linalg.norm(
            self.agent_pos[:, None, :] - self.agent_pos[None, :, :], axis=-1
        )
        collisions = np.triu(pairwise < 0.10, k=1).sum()
        reward -= float(collisions) * self.config.collision_penalty
        base = self._result()
        truncated = self._step >= self.episode_limit
        return StepResult(
            **base.__dict__, reward=reward, terminated=False, truncated=truncated,
            info={"collisions": int(collisions)},
        )


class PredatorPreyEnv(_ParticleBase):
    """Cooperative predators with scripted, evasive prey and local observations."""

    def __init__(self, config: EnvironmentConfig, seed: int | None = None) -> None:
        super().__init__(config, seed)
        self.n_prey = config.n_targets
        self.prey_pos = np.zeros((self.n_prey, 2), dtype=np.float32)
        self.prey_vel = np.zeros_like(self.prey_pos)
        self.obs_dim = 4 + 5 * self.n_prey + 5 * (self.n_agents - 1)
        self.state_dim = 4 * self.n_agents + 4 * self.n_prey

    def reset(self, seed: int | None = None) -> ResetResult:
        self._seed(seed)
        self._step = 0
        self.agent_pos = self._spawn(self.n_agents)
        self.agent_vel.fill(0.0)
        self.prey_pos = self._spawn(self.n_prey)
        self.prey_vel.fill(0.0)
        return self._result()

    def _move_prey(self) -> None:
        for p in range(self.n_prey):
            offsets = self.prey_pos[p] - self.agent_pos
            nearest = int(np.argmin(np.linalg.norm(offsets, axis=1)))
            direction = offsets[nearest]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = self._rng.normal(size=2).astype(np.float32)
                norm = float(np.linalg.norm(direction))
            noise = self._rng.normal(0.0, 0.20, size=2).astype(np.float32)
            desired = direction / max(norm, 1e-6) + noise
            desired /= max(float(np.linalg.norm(desired)), 1e-6)
            self.prey_vel[p] = 0.3 * self.prey_vel[p] + 0.7 * desired * self.config.prey_speed
        self.prey_pos = np.clip(
            self.prey_pos + self.prey_vel,
            -self.config.world_size,
            self.config.world_size,
        ).astype(np.float32)

    def _observations(self) -> np.ndarray:
        observations: list[np.ndarray] = []
        for i in range(self.n_agents):
            features: list[float] = [*self.agent_vel[i], *self.agent_pos[i]]
            for p in range(self.n_prey):
                relative = self.prey_pos[p] - self.agent_pos[i]
                visible = float(np.linalg.norm(relative) <= self.config.vision_radius)
                features.extend((relative * visible).tolist())
                features.extend((self.prey_vel[p] * visible).tolist())
                features.append(visible)
            for j in range(self.n_agents):
                if i == j:
                    continue
                relative = self.agent_pos[j] - self.agent_pos[i]
                visible = float(np.linalg.norm(relative) <= self.config.vision_radius)
                features.extend((relative * visible).tolist())
                features.extend((self.agent_vel[j] * visible).tolist())
                features.append(visible)
            observations.append(np.asarray(features, dtype=np.float32))
        return np.stack(observations)

    def _state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.agent_pos.reshape(-1), self.agent_vel.reshape(-1),
                self.prey_pos.reshape(-1), self.prey_vel.reshape(-1),
            ]
        ).astype(np.float32)

    def _result(self) -> ResetResult:
        return ResetResult(
            observations=self._observations(), state=self._state(),
            available_actions=self._available_actions(), adjacency=self._adjacency(),
        )

    def step(self, actions: np.ndarray) -> StepResult:
        self._move_agents(actions)
        self._move_prey()
        self._step += 1
        distances = np.linalg.norm(
            self.prey_pos[:, None, :] - self.agent_pos[None, :, :], axis=-1
        )
        closest = distances.min(axis=1)
        captures = closest <= self.config.capture_radius
        reward = -float(closest.sum()) + float(captures.sum()) * self.config.capture_reward
        if captures.any():
            self.prey_pos[captures] = self._spawn(int(captures.sum()))
            self.prey_vel[captures] = 0.0
        base = self._result()
        truncated = self._step >= self.episode_limit
        return StepResult(
            **base.__dict__, reward=reward, terminated=False, truncated=truncated,
            info={"captures": int(captures.sum())},
        )


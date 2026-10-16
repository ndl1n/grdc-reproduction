from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BufferShapes:
    capacity: int
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    hidden_dim: int


class RolloutBuffer:
    def __init__(self, shapes: BufferShapes) -> None:
        self.shapes = shapes
        t, n = shapes.capacity, shapes.n_agents
        self.observations = np.zeros((t, n, shapes.obs_dim), dtype=np.float32)
        self.states = np.zeros((t, shapes.state_dim), dtype=np.float32)
        self.adjacency = np.zeros((t, n, n), dtype=np.bool_)
        self.available_actions = np.zeros((t, n, shapes.n_actions), dtype=np.bool_)
        self.hidden = np.zeros((t, n, shapes.hidden_dim), dtype=np.float32)
        self.actions = np.zeros((t, n), dtype=np.int64)
        self.previous_actions = np.full((t, n), -1, dtype=np.int64)
        self.log_probabilities = np.zeros((t, n), dtype=np.float32)
        self.role_assignments = np.zeros((t, n), dtype=np.int64)
        self.role_probabilities: np.ndarray | None = None
        self.values = np.zeros(t, dtype=np.float32)
        self.next_values = np.zeros(t, dtype=np.float32)
        self.rewards = np.zeros(t, dtype=np.float32)
        self.episode_starts = np.zeros(t, dtype=np.bool_)
        self.episode_ends = np.zeros(t, dtype=np.bool_)
        self.advantages = np.zeros(t, dtype=np.float32)
        self.returns = np.zeros(t, dtype=np.float32)
        self.position = 0

    def reset(self, n_prototypes: int) -> None:
        self.position = 0
        self.role_probabilities = np.zeros(
            (self.shapes.capacity, self.shapes.n_agents, n_prototypes), dtype=np.float32
        )

    def add(
        self,
        *,
        observations: np.ndarray,
        state: np.ndarray,
        adjacency: np.ndarray,
        available_actions: np.ndarray,
        hidden: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        role_assignments: np.ndarray,
        role_probabilities: np.ndarray,
        value: float,
        next_value: float,
        reward: float,
        episode_start: bool,
        episode_end: bool,
        previous_actions: np.ndarray | None = None,
    ) -> None:
        if self.position >= self.shapes.capacity:
            raise RuntimeError("Rollout buffer is full")
        i = self.position
        self.observations[i] = observations
        self.states[i] = state
        self.adjacency[i] = adjacency
        self.available_actions[i] = available_actions
        self.hidden[i] = hidden
        self.actions[i] = actions
        self.previous_actions[i] = -1 if previous_actions is None else previous_actions
        self.log_probabilities[i] = log_probabilities
        self.role_assignments[i] = role_assignments
        assert self.role_probabilities is not None
        self.role_probabilities[i] = role_probabilities
        self.values[i] = value
        self.next_values[i] = next_value
        self.rewards[i] = reward
        self.episode_starts[i] = episode_start
        self.episode_ends[i] = episode_end
        self.position += 1

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> None:
        if self.position != self.shapes.capacity:
            raise RuntimeError("Buffer must be full before computing advantages")
        last_gae = 0.0
        for step in range(self.position - 1, -1, -1):
            delta = (
                self.rewards[step]
                + gamma * self.next_values[step]
                - self.values[step]
            )
            continuation = 1.0 - float(self.episode_ends[step])
            last_gae = delta + gamma * gae_lambda * continuation * last_gae
            self.advantages[step] = last_gae
        self.returns[:] = self.advantages + self.values

    def prototype_utilization(self) -> np.ndarray:
        assert self.role_probabilities is not None
        return self.role_probabilities.mean(axis=(0, 1))


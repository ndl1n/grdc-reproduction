from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from grdc.config import EnvironmentConfig


@dataclass
class ResetResult:
    observations: np.ndarray
    state: np.ndarray
    available_actions: np.ndarray
    adjacency: np.ndarray


@dataclass
class StepResult(ResetResult):
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MultiAgentEnv(ABC):
    n_agents: int
    obs_dim: int
    state_dim: int
    n_actions: int
    episode_limit: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> ResetResult:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: np.ndarray) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


def make_env(config: EnvironmentConfig, seed: int | None = None) -> MultiAgentEnv:
    if config.name == "cooperative_navigation":
        from .particle import CooperativeNavigationEnv

        return CooperativeNavigationEnv(config, seed=seed)
    if config.name == "predator_prey":
        from .particle import PredatorPreyEnv

        return PredatorPreyEnv(config, seed=seed)
    if config.name == "smacv2":
        from .smacv2_adapter import SMACv2Adapter

        return SMACv2Adapter(config, seed=seed)
    raise ValueError(f"Unsupported environment: {config.name}")


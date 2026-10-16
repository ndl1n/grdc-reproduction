from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvironmentConfig:
    name: str = "cooperative_navigation"
    n_agents: int = 5
    n_targets: int = 5
    episode_limit: int = 50
    vision_radius: float = 0.3
    world_size: float = 1.0
    agent_speed: float = 0.10
    prey_speed: float = 0.075
    collision_penalty: float = 1.0
    capture_radius: float = 0.10
    capture_reward: float = 0.0
    race: str = "protoss"
    n_enemies: int = 5
    sc2_path: str | None = None


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    attention_dim: int = 64
    n_prototypes: int = 10
    min_prototypes: int = 2
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.10
    orthogonal_init: bool = True
    use_roles: bool = True
    use_local_graph: bool = True
    use_intra_role_communication: bool = True
    # False preserves architecture/equations of pre-audit checkpoints.
    paper_equations: bool = False
    include_previous_action: bool = False
    stable_role_attention: bool = True


@dataclass
class TrainingConfig:
    total_steps: int = 10_000_000
    rollout_steps: int = 1024
    update_epochs: int = 10
    minibatch_chunks: int = 8
    recurrent_chunk_length: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.20
    value_clip: float = 0.20
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 10.0
    value_coef: float = 1.0
    policy_entropy_coef: float = 0.01
    role_entropy_coef: float = 0.01
    prototype_decorrelation_coef: float = 0.01
    pruning_sensitivity: float = 0.01
    pruning_warmup_steps: int = 500_000
    pruning_interval_steps: int = 250_000
    enable_pruning: bool = True
    eval_interval_steps: int = 100_000
    eval_episodes: int = 32
    save_interval_steps: int = 250_000
    normalize_advantages: bool = True
    critic_objective: str = "ppo_clipped"
    keep_step_checkpoints: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "grdc_cn"
    seed: int = 1
    device: str = "auto"
    output_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.environment.name not in {
            "cooperative_navigation",
            "predator_prey",
            "smacv2",
        }:
            raise ValueError(f"Unsupported environment: {self.environment.name}")
        if self.environment.n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if self.model.n_prototypes < self.model.min_prototypes:
            raise ValueError("n_prototypes must be >= min_prototypes")
        if self.model.min_prototypes < 1:
            raise ValueError("min_prototypes must be positive")
        if self.model.paper_equations and not self.model.stable_role_attention:
            raise ValueError("paper_equations mode requires stable_role_attention")
        if self.training.critic_objective not in {"ppo_clipped", "td"}:
            raise ValueError("critic_objective must be ppo_clipped or td")
        if self.training.eval_interval_steps < 1 or self.training.save_interval_steps < 1:
            raise ValueError("Evaluation/checkpoint intervals must be positive")
        if self.training.eval_episodes < 1 or self.training.total_steps < 1:
            raise ValueError("Evaluation episodes and total_steps must be positive")
        if not 0.0 < self.training.pruning_sensitivity < 1.0:
            raise ValueError("pruning_sensitivity must be in (0, 1)")
        if self.training.rollout_steps < self.training.recurrent_chunk_length:
            raise ValueError("rollout_steps must be >= recurrent_chunk_length")
        if self.training.rollout_steps % self.training.recurrent_chunk_length != 0:
            raise ValueError("rollout_steps must be divisible by recurrent_chunk_length")
        if self.training.minibatch_chunks < 1 or self.training.update_epochs < 1:
            raise ValueError("minibatch_chunks and update_epochs must be positive")
        if self.environment.name == "smacv2":
            if self.environment.race not in {"protoss", "terran", "zerg"}:
                raise ValueError("SMACv2 race must be protoss, terran, or zerg")
            if self.environment.n_agents not in {5, 10, 20}:
                raise ValueError("Paper scenarios use 5, 10, or 20 allied units")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(data: dict[str, Any]) -> ExperimentConfig:
    known = {"experiment_name", "seed", "device", "output_dir", "checkpoint_dir"}
    unknown = set(data) - known - {"environment", "model", "training"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    model_data = dict(data.get("model", {}))
    # Serialized configs/checkpoints predating the fix must retain original
    # floating-point forward behavior for retrospective evaluation.
    model_data.setdefault("stable_role_attention", bool(model_data.get("paper_equations", False)))
    cfg = ExperimentConfig(
        **{k: data[k] for k in known if k in data},
        environment=EnvironmentConfig(**data.get("environment", {})),
        model=ModelConfig(**model_data),
        training=TrainingConfig(**data.get("training", {})),
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _construct(data)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _construct(data)

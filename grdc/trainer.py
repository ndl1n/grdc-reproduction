from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from grdc.buffer import BufferShapes, RolloutBuffer
from grdc.config import ExperimentConfig
from grdc.envs import MultiAgentEnv, make_env
from grdc.models import CentralizedCritic, GRDCActor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@dataclass
class TrainMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    action_entropy: float = 0.0
    role_entropy: float = 0.0
    decorrelation_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._header_written = path.exists() and path.stat().st_size > 0

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class GRDCTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        seed_everything(config.seed)
        self.device = select_device(config.device)
        self.env = make_env(config.environment, seed=config.seed)
        self.actor = GRDCActor(self.env.obs_dim, self.env.n_actions, config.model).to(self.device)
        self.critic = CentralizedCritic(
            self.env.state_dim, config.model.hidden_dim, config.model.orthogonal_init
        ).to(self.device)
        self.actor_optimizer = self._make_actor_optimizer()
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.training.critic_lr, eps=config.training.adam_eps
        )
        self.buffer = RolloutBuffer(
            BufferShapes(
                capacity=config.training.rollout_steps,
                n_agents=self.env.n_agents,
                obs_dim=self.env.obs_dim,
                state_dim=self.env.state_dim,
                n_actions=self.env.n_actions,
                hidden_dim=config.model.hidden_dim,
            )
        )
        run_name = f"{config.experiment_name}_seed{config.seed}"
        self.run_dir = Path(config.output_dir) / run_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)
        metric_fields = [
            "step", "elapsed_seconds", "temperature", "active_prototypes",
            "rollout_return", "rollout_episode_length", "rollout_win_rate",
            "completed_episodes", "policy_loss", "value_loss", "action_entropy",
            "role_entropy", "decorrelation_loss", "approx_kl", "clip_fraction",
            "pruned_prototypes", "eval_return_mean", "eval_return_std",
            "eval_episode_length", "eval_win_rate",
        ] + [f"role_{i}_count" for i in range(config.model.n_prototypes)]
        self.csv = CsvLogger(self.run_dir / "metrics.csv", metric_fields)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.run_dir / "tensorboard")
        except ImportError:
            self.writer = None
        self.global_step = 0
        self.update_number = 0
        self.last_prune_step = 0
        self.best_eval_return = -float("inf")

    def _make_actor_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.training.actor_lr,
            eps=self.config.training.adam_eps,
        )

    def temperature(self) -> float:
        fraction = min(self.global_step / max(self.config.training.total_steps, 1), 1.0)
        start = self.config.model.gumbel_temperature_start
        end = self.config.model.gumbel_temperature_end
        return start + fraction * (end - start)

    @torch.no_grad()
    def collect_rollout(self) -> dict[str, float]:
        self.actor.eval()
        self.critic.eval()
        self.buffer.reset(self.config.model.n_prototypes)
        transition = self.env.reset(seed=self.config.seed + self.update_number)
        hidden = self.actor.initial_hidden(1, self.env.n_agents, self.device)
        episode_start = True
        episode_return = 0.0
        episode_length = 0
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        wins: list[float] = []

        for _ in range(self.config.training.rollout_steps):
            observations = torch.as_tensor(
                transition.observations, device=self.device
            ).unsqueeze(0)
            adjacency = torch.as_tensor(transition.adjacency, device=self.device).unsqueeze(0)
            available = torch.as_tensor(
                transition.available_actions, device=self.device
            ).unsqueeze(0)
            state = torch.as_tensor(transition.state, device=self.device).unsqueeze(0)
            actor_out = self.actor.forward_step(
                observations, hidden, adjacency, available,
                temperature=self.temperature(), deterministic=False,
            )
            distribution = Categorical(logits=actor_out.logits)
            actions = distribution.sample()
            log_probabilities = distribution.log_prob(actions)
            value = self.critic(state)

            next_transition = self.env.step(actions.squeeze(0).cpu().numpy())
            episode_end = next_transition.terminated or next_transition.truncated
            if next_transition.terminated:
                next_value = torch.zeros(1, device=self.device)
            else:
                next_state_tensor = torch.as_tensor(
                    next_transition.state, device=self.device
                ).unsqueeze(0)
                next_value = self.critic(next_state_tensor)

            self.buffer.add(
                observations=transition.observations,
                state=transition.state,
                adjacency=transition.adjacency,
                available_actions=transition.available_actions,
                hidden=hidden.squeeze(0).cpu().numpy(),
                actions=actions.squeeze(0).cpu().numpy(),
                log_probabilities=log_probabilities.squeeze(0).cpu().numpy(),
                role_assignments=actor_out.role_assignments.squeeze(0).cpu().numpy(),
                role_probabilities=actor_out.role_probabilities.squeeze(0).cpu().numpy(),
                value=float(value.item()),
                next_value=float(next_value.item()),
                reward=next_transition.reward,
                episode_start=episode_start,
                episode_end=episode_end,
            )
            self.global_step += 1
            episode_return += next_transition.reward
            episode_length += 1
            hidden = actor_out.hidden
            transition = next_transition
            episode_start = False
            if episode_end:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                if "battle_won" in next_transition.info:
                    wins.append(float(next_transition.info["battle_won"]))
                transition = self.env.reset()
                hidden = self.actor.initial_hidden(1, self.env.n_agents, self.device)
                episode_return = 0.0
                episode_length = 0
                episode_start = True

        self.buffer.compute_returns_and_advantages(
            self.config.training.gamma, self.config.training.gae_lambda
        )
        return {
            "rollout_return": float(np.mean(completed_returns)) if completed_returns else episode_return,
            "rollout_episode_length": float(np.mean(completed_lengths)) if completed_lengths else episode_length,
            "rollout_win_rate": float(np.mean(wins)) if wins else float("nan"),
            "completed_episodes": float(len(completed_returns)),
        }

    def _tensor(self, array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device, dtype=dtype)

    def update(self) -> TrainMetrics:
        self.actor.train()
        self.critic.train()
        cfg = self.config.training
        sequence_length = cfg.recurrent_chunk_length
        if self.buffer.position % sequence_length != 0:
            raise ValueError("rollout_steps must be divisible by recurrent_chunk_length")
        starts = np.arange(0, self.buffer.position, sequence_length)
        advantages = self.buffer.advantages.copy()
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = TrainMetrics()
        batches = 0
        for _ in range(cfg.update_epochs):
            np.random.shuffle(starts)
            for offset in range(0, len(starts), cfg.minibatch_chunks):
                batch_starts = starts[offset : offset + cfg.minibatch_chunks]
                if len(batch_starts) == 0:
                    continue
                indices = batch_starts[:, None] + np.arange(sequence_length)[None, :]
                # Time-major: [L, B, N, ...]
                idx = indices.T
                observations = self._tensor(self.buffer.observations[idx])
                adjacency = self._tensor(self.buffer.adjacency[idx])
                available = self._tensor(self.buffer.available_actions[idx])
                actions = self._tensor(self.buffer.actions[idx], torch.long)
                forced_roles = self._tensor(self.buffer.role_assignments[idx], torch.long)
                old_log_probabilities = self._tensor(self.buffer.log_probabilities[idx])
                episode_starts = self._tensor(self.buffer.episode_starts[idx], torch.bool)
                hidden = self._tensor(self.buffer.hidden[batch_starts])

                new_log_probs: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                role_probabilities: list[torch.Tensor] = []
                for time_index in range(sequence_length):
                    if time_index > 0:
                        keep = (~episode_starts[time_index]).to(hidden.dtype).view(-1, 1, 1)
                        hidden = hidden * keep
                    actor_out = self.actor.forward_step(
                        observations[time_index], hidden, adjacency[time_index],
                        available[time_index], temperature=self.temperature(),
                        deterministic=False, forced_roles=forced_roles[time_index],
                    )
                    distribution = Categorical(logits=actor_out.logits)
                    new_log_probs.append(distribution.log_prob(actions[time_index]))
                    entropies.append(distribution.entropy())
                    role_probabilities.append(actor_out.role_probabilities)
                    hidden = actor_out.hidden

                new_log_probabilities = torch.stack(new_log_probs)
                action_entropy = torch.stack(entropies).mean()
                role_probs = torch.stack(role_probabilities)
                ratio = torch.exp(new_log_probabilities - old_log_probabilities)
                batch_advantages = self._tensor(advantages[idx]).unsqueeze(-1)
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(
                    ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip
                ) * batch_advantages
                policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
                role_entropy = self.actor.role_entropy(role_probs)
                decorrelation = self.actor.prototype_decorrelation_loss()
                actor_loss = (
                    policy_loss
                    - cfg.policy_entropy_coef * action_entropy
                    - cfg.role_entropy_coef * role_entropy
                    + cfg.prototype_decorrelation_coef * decorrelation
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                states = self._tensor(self.buffer.states[idx]).reshape(-1, self.env.state_dim)
                returns = self._tensor(self.buffer.returns[idx]).reshape(-1)
                old_values = self._tensor(self.buffer.values[idx]).reshape(-1)
                values = self.critic(states)
                clipped_values = old_values + (values - old_values).clamp(
                    -cfg.value_clip, cfg.value_clip
                )
                value_loss_unclipped = (values - returns).pow(2)
                value_loss_clipped = (clipped_values - returns).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probabilities - old_log_probabilities
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > cfg.ppo_clip).float().mean()
                totals.policy_loss += float(policy_loss.item())
                totals.value_loss += float(value_loss.item())
                totals.action_entropy += float(action_entropy.item())
                totals.role_entropy += float(role_entropy.item())
                totals.decorrelation_loss += float(decorrelation.item())
                totals.approx_kl += float(approx_kl.item())
                totals.clip_fraction += float(clip_fraction.item())
                batches += 1

        for field_name in totals.__dataclass_fields__:
            setattr(totals, field_name, getattr(totals, field_name) / max(batches, 1))
        self.update_number += 1
        return totals

    def maybe_prune(self) -> list[int]:
        cfg = self.config.training
        if not cfg.enable_pruning or not self.config.model.use_roles:
            return []
        if self.global_step < cfg.pruning_warmup_steps:
            return []
        if self.global_step - self.last_prune_step < cfg.pruning_interval_steps:
            return []
        utilization = self._tensor(self.buffer.prototype_utilization())
        pruned = self.actor.prune(utilization, cfg.pruning_sensitivity)
        self.last_prune_step = self.global_step
        if pruned:
            # The paper explicitly reinitializes optimizer state after pruning.
            self.actor_optimizer = self._make_actor_optimizer()
        return pruned

    @torch.no_grad()
    def evaluate(self, episodes: int | None = None) -> dict[str, float]:
        self.actor.eval()
        self.critic.eval()
        eval_env = make_env(self.config.environment, seed=self.config.seed + 100_000)
        returns: list[float] = []
        lengths: list[int] = []
        wins: list[float] = []
        role_counts = np.zeros(self.config.model.n_prototypes, dtype=np.int64)
        try:
            for episode in range(episodes or self.config.training.eval_episodes):
                transition = eval_env.reset(seed=self.config.seed + 100_000 + episode)
                hidden = self.actor.initial_hidden(1, eval_env.n_agents, self.device)
                episode_return = 0.0
                length = 0
                while True:
                    out = self.actor.forward_step(
                        self._tensor(transition.observations).unsqueeze(0), hidden,
                        self._tensor(transition.adjacency).unsqueeze(0),
                        self._tensor(transition.available_actions).unsqueeze(0),
                        temperature=self.temperature(), deterministic=True,
                    )
                    actions = out.logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                    roles = out.role_assignments.squeeze(0).cpu().numpy()
                    role_counts += np.bincount(
                        roles, minlength=self.config.model.n_prototypes
                    )
                    hidden = out.hidden
                    transition = eval_env.step(actions)
                    episode_return += transition.reward
                    length += 1
                    if transition.terminated or transition.truncated:
                        if "battle_won" in transition.info:
                            wins.append(float(transition.info["battle_won"]))
                        break
                returns.append(episode_return)
                lengths.append(length)
        finally:
            eval_env.close()
        result = {
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_episode_length": float(np.mean(lengths)),
            "eval_win_rate": float(np.mean(wins)) if wins else float("nan"),
        }
        for role, count in enumerate(role_counts):
            result[f"role_{role}_count"] = float(count)
        return result

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "config": self.config.to_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "global_step": self.global_step,
                "update_number": self.update_number,
                "last_prune_step": self.last_prune_step,
                "best_eval_return": self.best_eval_return,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.update_number = int(checkpoint.get("update_number", 0))
        self.last_prune_step = int(checkpoint.get("last_prune_step", 0))
        self.best_eval_return = float(checkpoint.get("best_eval_return", -float("inf")))

    def train(self) -> None:
        cfg = self.config.training
        next_eval = max(cfg.eval_interval_steps, 1)
        next_save = max(cfg.save_interval_steps, 1)
        started = time.time()
        try:
            while self.global_step < cfg.total_steps:
                rollout_metrics = self.collect_rollout()
                train_metrics = self.update()
                pruned = self.maybe_prune()
                row: dict[str, Any] = {
                    "step": self.global_step,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "temperature": self.temperature(),
                    "active_prototypes": self.actor.n_active_prototypes,
                    **rollout_metrics,
                    **train_metrics.__dict__,
                }
                if pruned:
                    row["pruned_prototypes"] = ";".join(map(str, pruned))

                if self.global_step >= next_eval:
                    evaluation = self.evaluate()
                    row.update(evaluation)
                    next_eval += cfg.eval_interval_steps
                    if evaluation["eval_return_mean"] > self.best_eval_return:
                        self.best_eval_return = evaluation["eval_return_mean"]
                        self.save_checkpoint("best.pt")
                if self.global_step >= next_save:
                    self.save_checkpoint(f"step_{self.global_step}.pt")
                    next_save += cfg.save_interval_steps

                self.csv.write(row)
                if self.writer is not None:
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            self.writer.add_scalar(key, value, self.global_step)
                print(
                    f"step={self.global_step} return={rollout_metrics['rollout_return']:.3f} "
                    f"policy={train_metrics.policy_loss:.4f} value={train_metrics.value_loss:.4f} "
                    f"roles={self.actor.n_active_prototypes}"
                )
            self.save_checkpoint("last.pt")
        finally:
            self.env.close()
            if self.writer is not None:
                self.writer.close()

"""Bounded diagnostic runs, isolated from research results. Not a reproduction score."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import types

import torch

from grdc.config import load_config, ModelConfig
from grdc.models import GRDCActor
from grdc.trainer import GRDCTrainer


def legacy_message(self, hidden, vectors, alive):
    scores = self.role_query(hidden) @ self.role_key(hidden).transpose(-1, -2)
    scores = scores / math.sqrt(self.attention_dim)
    scores = scores - scores.amax(-1, keepdim=True).detach()
    affinity = vectors @ vectors.transpose(-1, -2)
    peer = ~torch.eye(hidden.shape[1], dtype=torch.bool)[None] & alive[:, None] & alive[:, :, None]
    numerator = scores.exp() * affinity.clamp_min(0) * peer
    weights = numerator / numerator.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights @ self.role_value(hidden)


def gradient_probe(legacy):
    torch.manual_seed(1)
    actor = GRDCActor(39, 5, ModelConfig())
    if legacy:
        actor._intra_role_message = types.MethodType(legacy_message, actor)
    output = actor.forward_step(
        torch.randn(1, 5, 39), actor.initial_hidden(1, 5, torch.device("cpu")),
        torch.ones(1, 5, 5, dtype=torch.bool), torch.ones(1, 5, 5, dtype=torch.bool),
        temperature=1., forced_roles=torch.tensor([[0, 1, 2, 3, 4]]))
    (-output.logits.log_softmax(-1)[0, :, 0].mean()).backward()
    norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 10, error_if_nonfinite=True))
    return {"pre_clip_actor_grad_norm": norm,
            "post_clip_policy_grad_norm": actor.policy[-1].weight.grad.norm().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision_validation"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {"purpose": "smoke/gradient validation, not evidence of winning or reproduction",
               "threads": 1, "gradient_before": gradient_probe(True), "gradient_after": gradient_probe(False),
               "short_runs": []}
    for filename in ["cooperative_navigation.json", "mappo_cooperative_navigation.json"]:
        cfg = load_config(Path("configs/revised") / filename)
        cfg.device = "cpu"
        cfg.output_dir = str(output / "runs")
        cfg.checkpoint_dir = str(output / "checkpoints")
        run = Path(cfg.output_dir) / f"{cfg.experiment_name}_seed{cfg.seed}"
        if (run / "metrics.csv").exists():
            raise RuntimeError(f"Choose a new --output-dir; refusing to append to {run}")
        trainer = GRDCTrainer(cfg)
        before = trainer.actor.policy[-1].weight.detach().clone()
        started = time.perf_counter()
        trainer.train(stop_after_steps=args.steps)
        elapsed = time.perf_counter() - started
        # Standard final evaluation is an integrity check, not a held-out paper result.
        metrics = trainer.evaluate(32)
        with (run / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        results["short_runs"].append({
            "experiment": cfg.experiment_name, "actual_steps": trainer.global_step,
            "configured_horizon": cfg.training.total_steps, "temperature": trainer.temperature(),
            "seconds": elapsed, "steps_per_second": trainer.global_step / elapsed,
            "max_actor_grad_norm": max(float(r["actor_grad_norm_max"]) for r in rows),
            "policy_head_changed": not torch.equal(before, trainer.actor.policy[-1].weight),
            "eval_episodes": 32, "eval_mean_step_reward": metrics["eval_return_mean"] / metrics["eval_episode_length"]})
    (output / "validation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

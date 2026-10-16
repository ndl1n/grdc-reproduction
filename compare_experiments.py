"""Read-only analysis of CN runs; writes derived results, never source runs/checkpoints.

Usage: python compare_experiments.py --output-dir results/cn_comparison
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grdc.config import config_from_dict, load_config
from grdc.envs import make_env
from grdc.models import GRDCActor


COLORS = {"GRDC": "#2563a6", "MAPPO": "#d66a20"}
METHODS = ("GRDC", "MAPPO")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_history(path):
    """Infer final chronological branch: a later rewind supersedes earlier tail.

    This is a documented inference, not recovery of unrecorded checkpoint lineage.
    Do not merely sort/drop duplicate steps: that can retain abandoned branches.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    kept, rewinds = [], []
    for line, row in enumerate(raw, 2):
        step = int(row["step"])
        if kept and step <= int(kept[-1]["step"]):
            rewinds.append({"csv_line": line, "from_step": int(kept[-1]["step"]),
                            "restart_first_logged_step": step})
        while kept and int(kept[-1]["step"]) >= step:
            kept.pop()
        kept.append(row)
    return kept, {"raw_rows": len(raw), "retained_rows": len(kept),
                  "superseded_rows": len(raw) - len(kept), "rewinds": rewinds}


def number(row, key):
    value = row.get(key, "")
    return float(value) if value else float("nan")


def config_diff(left, right, prefix=""):
    result = []
    for key in sorted(left.keys() | right.keys()):
        name = f"{prefix}.{key}" if prefix else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            result.extend(config_diff(a, b, name))
        elif a != b:
            result.append({"field": name, "left": a, "right": b})
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    if cfg.environment.name != "cooperative_navigation":
        raise ValueError("This analysis requires Cooperative Navigation")
    torch.manual_seed(cfg.seed)
    env = make_env(cfg.environment, seed=cfg.seed)
    try:
        actor = GRDCActor(env.obs_dim, env.n_actions, cfg.model)
        actor.load_state_dict(payload["actor"])
        actor.eval()
    finally:
        env.close()
    return actor, cfg, payload


@torch.inference_mode()
def evaluate(actor, cfg, step, episodes, first_scene_seed):
    env = make_env(cfg.environment, seed=first_scene_seed)
    fraction = min(step / max(cfg.training.total_steps, 1), 1.0)
    temperature = cfg.model.gumbel_temperature_start + fraction * (
        cfg.model.gumbel_temperature_end - cfg.model.gumbel_temperature_start)
    records, counts = [], np.zeros(cfg.model.n_prototypes, dtype=np.int64)
    try:
        for episode in range(episodes):
            scene_seed = first_scene_seed + episode
            transition = env.reset(seed=scene_seed)
            hidden = actor.initial_hidden(1, env.n_agents, torch.device("cpu"))
            previous_actions = torch.full((1, env.n_agents), -1, dtype=torch.long)
            total, collision_pairs, distance_sum, length = 0.0, 0, 0.0, 0
            while True:
                output = actor.forward_step(
                    torch.as_tensor(transition.observations, dtype=torch.float32)[None], hidden,
                    torch.as_tensor(transition.adjacency)[None],
                    torch.as_tensor(transition.available_actions)[None],
                    temperature=temperature, deterministic=True, previous_actions=previous_actions)
                action = output.logits.argmax(-1).squeeze(0).numpy()
                previous_actions = torch.as_tensor(action).unsqueeze(0)
                if cfg.model.use_roles:
                    counts += np.bincount(output.role_assignments.squeeze(0).numpy(),
                                          minlength=cfg.model.n_prototypes)
                hidden = output.hidden
                transition = env.step(action)
                total += transition.reward
                collision_pairs += transition.info["collisions"]
                distances = np.linalg.norm(env.landmark_pos[:, None] - env.agent_pos[None], axis=-1)
                distance_sum += float(distances.min(axis=1).sum())
                length += 1
                if transition.terminated or transition.truncated:
                    break
            records.append({"scene_seed": scene_seed, "episode_return": total,
                            "episode_length": length, "mean_step_reward": total / length,
                            "collision_pairs_per_step": collision_pairs / length,
                            "sum_landmark_nearest_distance_per_step": distance_sum / length})
    finally:
        env.close()
    return records, counts


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def format_axis(ax, title, ylabel, xlabel="Environment steps (millions)"):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cn_comparison"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--scene-seed", type=int, default=900000)
    args = parser.parse_args()
    if args.episodes < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Need at least 2 evaluation episodes and unique training seeds")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() in (args.runs_dir.resolve(), args.checkpoints_dir.resolve()):
        raise ValueError("Output must not overwrite source directories")
    torch.set_num_threads(1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "figure.facecolor": "white"})
    histories, configs, audit, metadata = {}, {}, {}, {}
    summaries, episode_rows, role_rows, manifests, source_rows = [], [], [], [], []
    for method in METHODS:
        for seed in args.seeds:
            name = f"{method.lower()}_cooperative_navigation_seed{seed}"
            folder = args.runs_dir / name
            history, info = clean_history(folder / "metrics.csv")
            cfg = load_config(folder / "resolved_config.json")
            checkpoint = args.checkpoints_dir / name / "last.pt"
            actor, checkpoint_cfg, payload = load_model(checkpoint)
            step = int(payload["global_step"])
            if checkpoint_cfg.seed != seed or cfg.seed != seed:
                raise ValueError(f"Seed mismatch: {name}")
            if step < cfg.training.total_steps or step != int(history[-1]["step"]):
                raise ValueError(f"Final checkpoint not complete/aligned with CSV: {name}")
            # Preserve both sources; mismatches in substantive configuration are fatal.
            harmless = {"device", "output_dir", "checkpoint_dir"}
            differences = config_diff(cfg.to_dict(), checkpoint_cfg.to_dict())
            if any(d["field"] not in harmless for d in differences):
                raise ValueError(f"Checkpoint/run config disagreement: {name}: {differences}")
            info["checkpoint_config_differences"] = differences
            stride, interval = cfg.training.rollout_steps, cfg.training.eval_interval_steps
            targets = set(math.ceil(t / stride) * stride
                          for t in range(interval, cfg.training.total_steps + 1, interval))
            evals = [r for r in history if math.isfinite(number(r, "eval_return_mean"))]
            canonical = [r for r in evals if int(r["step"]) in targets]
            if not canonical or len({int(r["step"]) for r in canonical}) != len(canonical):
                raise ValueError(f"Invalid evaluation history: {name}")
            if any(number(r, "eval_episode_length") != cfg.environment.episode_limit for r in evals):
                raise ValueError("Per-step conversion requires fixed episode lengths")
            info.update(eval_rows=len(evals), canonical_eval_rows=len(canonical),
                        missing_canonical_steps=sorted(targets - {int(r["step"]) for r in canonical}))
            histories[method, seed] = (history, canonical)
            configs[method, seed] = cfg.to_dict()
            audit[name] = info
            write_csv(out / f"{name}_cleaned.csv", history)
            # Reproduce the logged final evaluation before using a new scene set.
            validation, _ = evaluate(actor, cfg, step, cfg.training.eval_episodes, seed + 100000)
            logged_final = evals[-1]
            reproduced = mean(r["episode_return"] for r in validation)
            info["reproduced_logged_final_return"] = reproduced
            info["logged_final_return_difference"] = reproduced - number(logged_final, "eval_return_mean")
            info["logged_final_reproduction_close"] = math.isclose(
                reproduced, number(logged_final, "eval_return_mean"), abs_tol=1e-4)
            episodes, counts = evaluate(actor, cfg, step, args.episodes, args.scene_seed)
            for r in episodes:
                episode_rows.append({"method": method, "training_seed": seed, **r})
            final = evals[-1]
            xs = np.array([int(r["step"]) for r in canonical], float)
            ys = np.array([number(r, "eval_return_mean") / number(r, "eval_episode_length")
                           for r in canonical])
            trapezoid = getattr(np, "trapezoid", np.trapz)
            auc = float(trapezoid(ys, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else float("nan")
            values = [r["mean_step_reward"] for r in episodes]
            summary = {"method": method, "training_seed": seed, "final_step": step,
                       "heldout_episodes": args.episodes, "heldout_reward_mean": mean(values),
                       "heldout_reward_episode_std": stdev(values),
                       "heldout_reward_episode_sem": stdev(values) / math.sqrt(len(values)),
                       "heldout_episode_return_mean": mean(r["episode_return"] for r in episodes),
                       "collision_pairs_per_step": mean(r["collision_pairs_per_step"] for r in episodes),
                       "sum_landmark_nearest_distance_per_step": mean(
                           r["sum_landmark_nearest_distance_per_step"] for r in episodes),
                       "logged_final_eval_reward": number(final, "eval_return_mean") / number(final, "eval_episode_length"),
                       "logged_best_eval_reward": max(ys),
                       "logged_final_1m_eval_mean": mean(y for x, y in zip(xs, ys) if x > math.ceil((cfg.training.total_steps - 1e6) / stride) * stride),
                       "learning_curve_time_average": auc,
                       "curve_start_step": int(xs[0]), "curve_end_step": int(xs[-1]),
                       "active_roles_final": actor.n_active_prototypes if cfg.model.use_roles else "N/A",
                       "actor_total_parameters": sum(p.numel() for p in actor.parameters())}
            summaries.append(summary)
            if cfg.model.use_roles:
                for role, count in enumerate(counts):
                    role_rows.append({"method": method, "training_seed": seed,
                                      "role": role, "active": bool(actor.active_prototypes[role]),
                                      "assignments": int(count), "fraction": count / counts.sum()})
            best = torch.load(checkpoint.parent / "best.pt", map_location="cpu", weights_only=False)
            files = list(checkpoint.parent.glob("*.pt"))
            for f in files:
                if f.name in {"last.pt", "best.pt"}:
                    recommendation = "KEEP"
                    reason = "final full checkpoint" if f.name == "last.pt" else "best validation checkpoint"
                elif f.name.startswith("step_"):
                    recommendation, reason = "OPTIONAL_REMOVE", "intermediate snapshot; loses retrospective model analysis"
                else:
                    recommendation, reason = "REVIEW", "unrecognized checkpoint name"
                manifests.append({"run": name, "file": str(f.resolve()), "bytes": f.stat().st_size,
                                  "recommendation": recommendation, "reason": reason})
            metadata[name] = {"last_step": step, "best_step": int(best["global_step"]),
                              "best_eval_episode_return": best["best_eval_return"],
                              "checkpoint_count": len(files), "total_bytes": sum(f.stat().st_size for f in files),
                              "last_checkpoint_keys": list(payload),
                              "last_checkpoint_tensor_bytes": {
                                  k: tensor_bytes(v) for k, v in payload.items()},
                              "last_checkpoint_sha256": file_hash(checkpoint)}
            for src in (folder / "metrics.csv", folder / "resolved_config.json", checkpoint):
                source_rows.append({"run": name, "source": str(src.resolve()),
                                    "bytes": src.stat().st_size, "sha256": file_hash(src)})
            # Verify reward decomposition independently, using raw episode records.
            for r in episodes:
                expected = -(r["sum_landmark_nearest_distance_per_step"] +
                             cfg.environment.collision_penalty * r["collision_pairs_per_step"])
                if not math.isclose(r["mean_step_reward"], expected, abs_tol=1e-6):
                    raise AssertionError("Reward decomposition does not reconcile")
            print(f"{name}: last={step}, heldout={mean(values):.6f}, "
                  f"cleaned={info['superseded_rows']}, eval_grid={len(canonical)}", flush=True)

    # Compare environment/model/PPO settings across seeds and methods; record every difference.
    comparisons = {}
    reference = configs["GRDC", args.seeds[0]]
    for (method, seed), cfg in configs.items():
        comparisons[f"{method}_seed{seed}"] = config_diff(reference, cfg)
    audit["configuration_differences_vs_grdc_seed1"] = comparisons
    allowed = {"seed", "experiment_name", "device", "output_dir", "checkpoint_dir",
               "training.save_interval_steps", "model.use_roles", "model.use_local_graph",
               "model.use_intra_role_communication", "training.role_entropy_coef",
               "training.prototype_decorrelation_coef", "training.enable_pruning"}
    audit["unexpected_configuration_differences"] = [
        {"run": run, **d} for run, diffs in comparisons.items() for d in diffs if d["field"] not in allowed]
    metric_keys = ["heldout_reward_mean", "logged_final_eval_reward", "logged_final_1m_eval_mean",
                   "learning_curve_time_average", "collision_pairs_per_step",
                   "sum_landmark_nearest_distance_per_step"]
    method_summary = []
    for method in METHODS:
        group = [r for r in summaries if r["method"] == method]
        for metric in metric_keys:
            vals = [r[metric] for r in group]
            method_summary.append({"method": method, "metric": metric, "training_seeds": len(vals),
                                   "mean": mean(vals), "seed_sample_std": stdev(vals) if len(vals) > 1 else ""})
    pairs = []
    for seed in args.seeds:
        a, b = [next(r for r in summaries if r["method"] == m and r["training_seed"] == seed)
                for m in METHODS]
        pairs.append({"training_seed": seed, "grdc_reward": a["heldout_reward_mean"],
                      "mappo_reward": b["heldout_reward_mean"],
                      "grdc_minus_mappo": a["heldout_reward_mean"] - b["heldout_reward_mean"]})
    write_csv(out / "run_summary.csv", summaries)
    write_csv(out / "method_summary.csv", method_summary)
    write_csv(out / "paired_seed_comparison.csv", pairs)
    write_csv(out / "heldout_episodes.csv", episode_rows)
    write_csv(out / "role_utilization.csv", role_rows)
    write_csv(out / "checkpoint_retention.csv", manifests)
    write_csv(out / "sources.csv", source_rows)
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Figure 1: unsmoothed validation trajectories plus common-grid mean +/- seed SD.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for method in METHODS:
        groups = []
        for index, seed in enumerate(args.seeds):
            rows = histories[method, seed][1]
            x = np.array([int(r["step"]) for r in rows])
            y = np.array([number(r, "eval_return_mean") / number(r, "eval_episode_length") for r in rows])
            axes[0].plot(x / 1e6, y, color=COLORS[method], linestyle=["-", "--", ":", "-."][index % 4],
                         linewidth=1.3, label=f"{method} seed {seed}")
            groups.append(dict(zip(x, y)))
        common = sorted(set.intersection(*(set(g) for g in groups)))
        matrix = np.array([[g[t] for t in common] for g in groups])
        ymean = matrix.mean(axis=0)
        spread = matrix.std(axis=0, ddof=1) if len(groups) > 1 else np.zeros(len(common))
        x = np.array(common) / 1e6
        axes[1].plot(x, ymean, color=COLORS[method], label=method)
        axes[1].fill_between(x, ymean-spread, ymean+spread, color=COLORS[method], alpha=.16)
    format_axis(axes[0], "Individual training seeds (no smoothing)", "Evaluation mean step reward (higher is better)")
    format_axis(axes[1], "Mean +/- sample SD across training seeds", "Evaluation mean step reward")
    for ax in axes:
        ax.legend(fontsize=9)
    save_figure(fig, out, "01_learning_curves")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for seed in args.seeds:
        vals = [next(r["heldout_reward_mean"] for r in summaries if r["method"] == m and r["training_seed"] == seed) for m in METHODS]
        axes[0].plot([0, 1], vals, color=".7", linewidth=1, zorder=1)
        for i, (method, val) in enumerate(zip(METHODS, vals)):
            axes[0].scatter(i, val, color=COLORS[method], marker="o" if seed == args.seeds[0] else "s", s=65, zorder=3)
            axes[0].annotate(f"seed {seed}: {val:.3f}", (i, val), xytext=(9, 3), textcoords="offset points", fontsize=9)
    for i, method in enumerate(METHODS):
        vals = [r["heldout_reward_mean"] for r in summaries if r["method"] == method]
        axes[0].plot([i-.14, i+.14], [mean(vals)]*2, color=COLORS[method], linewidth=3)
    axes[0].set_xticks([0, 1], METHODS)
    axes[0].set_xlim(-.35, 1.65)
    format_axis(axes[0], f"Final models: {args.episodes} shared held-out scenes", "Mean step reward (higher is better)", "Method")
    for i, pair in enumerate(pairs):
        delta = pair["grdc_minus_mappo"]
        axes[1].bar(i, delta, width=.5, color=COLORS["GRDC"] if delta > 0 else COLORS["MAPPO"])
        axes[1].annotate(f"{delta:+.3f}", (i, delta), xytext=(0, 6 if delta > 0 else -15), textcoords="offset points", ha="center")
    axes[1].axhline(0, color=".3", linewidth=1)
    axes[1].set_xticks(range(len(pairs)), [f"seed {r['training_seed']}" for r in pairs])
    axes[1].margins(y=.25)
    format_axis(axes[1], "Paired difference: GRDC minus MAPPO", "Mean step reward difference", "Training seed (negative favors MAPPO)")
    save_figure(fig, out, "02_final_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, metric, title in zip(axes,
            ["sum_landmark_nearest_distance_per_step", "collision_pairs_per_step"],
            ["Landmark coverage cost (lower is better)", "Collision cost (lower is better)"]):
        for i, method in enumerate(METHODS):
            vals = [r[metric] for r in summaries if r["method"] == method]
            ax.bar(i, mean(vals), color=COLORS[method], alpha=.28, width=.5)
            for j, val in enumerate(vals):
                ax.scatter(i + (j-(len(vals)-1)/2)*.08, val, color=COLORS[method], marker="o" if j == 0 else "s", zorder=3)
        ax.set_xticks([0, 1], METHODS)
        format_axis(ax, title, "Sum of nearest distances per step" if metric.startswith("sum") else "Colliding agent pairs per step", "Method (points = training seeds)")
    save_figure(fig, out, "03_reward_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for j, seed in enumerate(args.seeds):
        rows = histories["GRDC", seed][0]
        axes[0].plot([int(r["step"])/1e6 for r in rows], [number(r, "active_prototypes") for r in rows],
                     linestyle=["-", "--", ":", "-."][j % 4], label=f"GRDC seed {seed}")
        role_group = [r for r in role_rows if r["training_seed"] == seed]
        axes[1].bar(np.arange(len(role_group)) + (j-(len(args.seeds)-1)/2)*.34,
                    [r["fraction"] for r in role_group], width=.32, label=f"seed {seed}")
    format_axis(axes[0], "Prototype pruning during training", "Active prototypes")
    format_axis(axes[1], "Role assignments on held-out scenes", "Fraction of agent decisions", "Role index (indices are not aligned across seeds)")
    axes[1].set_xticks(range(len(role_group)))
    for ax in axes:
        ax.legend()
    save_figure(fig, out, "04_role_diagnostics")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, metric, title in zip(axes.flat,
            ["action_entropy", "approx_kl", "clip_fraction", "value_loss"],
            ["Policy entropy", "Approximate KL", "PPO clip fraction", "Critic loss"]):
        for method in METHODS:
            for j, seed in enumerate(args.seeds):
                rows = histories[method, seed][0]
                x = np.array([int(r["step"]) for r in rows])
                y = np.array([number(r, metric) for r in rows])
                bins = (x - 1) // 100000
                bx, by = [], []
                for key in np.unique(bins):
                    selected = bins == key
                    bx.append(float(x[selected].mean()) / 1e6)
                    by.append(float(np.nanmean(y[selected])))
                ax.plot(bx, by, color=COLORS[method], linestyle=["-", "--", ":", "-."][j % 4],
                        label=f"{method} seed {seed}", linewidth=1.2)
        format_axis(ax, title + " (100k-step bins)", metric)
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, out, "05_optimization_diagnostics")

    total = sum(r["bytes"] for r in manifests)
    optional = sum(r["bytes"] for r in manifests if r["recommendation"] == "OPTIONAL_REMOVE")
    report = ["# Cooperative Navigation 實驗比較", "",
              "獨立重實作；GRDC 與 recurrent MAPPO；training seeds=" + str(args.seeds), "",
              "## 主要結果", "",
              f"使用最終 last.pt，在相同 {args.episodes} 個新場景（seeds {args.scene_seed}–{args.scene_seed+args.episodes-1}）以 deterministic argmax 評估。每回合固定 50 steps。",
              "每步平均團隊獎勵越高越好；沒有再除以 agent 數。不是訓練期間最高分挑選結果。", "",
              "| 方法 | Training seed | 最終步數 | 新場景每步獎勵 | Episode SD（每步單位） | 訓練最後一次評估 | 最後 1M 評估均值 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| {r['method']} | {r['training_seed']} | {r['final_step']} | {r['heldout_reward_mean']:.6f} | {r['heldout_reward_episode_std']:.6f} | {r['logged_final_eval_reward']:.6f} | {r['logged_final_1m_eval_mean']:.6f} |")
    report += ["", "| 方法 | 新場景平均 ± 跨 training seed 樣本 SD |", "|---|---:|"]
    for method in METHODS:
        r = next(r for r in method_summary if r["method"] == method and r["metric"] == "heldout_reward_mean")
        report.append(f"| {method} | {r['mean']:.6f} ± {r['seed_sample_std']:.6f} |")
    delta = mean(p["grdc_minus_mappo"] for p in pairs)
    report += ["", f"配對平均差（GRDC−MAPPO）={delta:+.6f} reward/step；正值支持 GRDC、負值支持 MAPPO。",
               "只有兩個 training seeds，這是描述性初步結果。不提供顯著性結論、跨 seed 95% CI 或把 128 episodes 當 128 training seeds。",
               "新場景在檢視 seed 1 舊結果後選定；未用新測試分數調參。若之後依此調參，應另留新的測試場景。", "",
               "## 如何解讀這次結果", "",
               "- 先看個別 seeds，再看方法平均。這次結果是此環境與此獨立實作的初步比較，不能泛化為原論文方法無效。",
               "- GRDC 原型數在約 0.5–0.75M 步降至最低值 2；不代表最終真的使用兩個均衡角色。",
               "- 新場景 deterministic 評估中，GRDC seed 1 有 98.525% 決策集中在一個角色，seed 2 為 99.9875%。這是角色使用高度集中的觀察，未證明角色崩塌是分數較低的因果來源。",
               "- GRDC seed 1 前段 action entropy 接近 log(5)，同時 KL 與 clip fraction 很低，提示策略更新值得檢查；需進一步核對梯度、角色與 PPO 實作，不能僅憑圖判定 bug。",
               "- 建議先凍結目前程式與結果，檢查實作和論文的差異，再做關閉剪枝／角色／通訊等固定預算消融；不要直接把不佳結果歸因於 seed 或只增加訓練步數。", "",
               "## 圖片用途", "",
               "1. 01_learning_curves：原始未平滑評估曲線；右圖陰影為跨 seed 樣本 SD，不是 episode SD 或 CI。",
               "2. 02_final_comparison：新場景最終模型比較；連線表示相同 training seed；短橫線表示方法平均。",
               "3. 03_reward_decomposition：目標距離與碰撞成本，均越低越好。獎勵 = −目標距離總和 − 碰撞懲罰。",
               "4. 04_role_diagnostics：GRDC 活躍原型與角色使用率。不同 seeds 的角色編號沒有語意對齊。MAPPO 的角色數不適用。",
               "5. 05_optimization_diagnostics：100k-step 區間平均的 PPO 診斷；loss 大小本身不能當作方法優劣依據。", "",
               "## 數據處理與可比性", "",
               "- 不更動來源 CSV、設定或 checkpoint。來源 SHA256 記錄於 sources.csv。",
               "- 依 CSV 寫入順序，遇到 step 回退，移除舊分支 >= 新 step 的尾端。這是推定續訓分支；原紀錄未保存完整 checkpoint lineage，無法證明位元級連續。",
               "- 主學習曲線只取每 100k steps 的評估門檻（向上對齊 1024 rollout steps），排除續訓排程補追造成的密集額外評估。",
               "- 共同步數才計算跨 seed 曲線平均，不外插；缺少的 canonical steps 列在 data_audit.json。",
               "- 最後 1M 均值為 (9M,10M] 的名目評估點（允許 rollout 對齊超出不到 1024 steps）；時間平均曲線為第一個到最後一個名目評估點的梯形積分除以步數範圍。",
               "- 不用最大分數或不同 checkpoint 預算混比。logged_best_eval_reward 只作輔助，使用相同評估網格。",
               "- 新評估使用 CPU、同一份目前程式和相同場景；不重新訓練。歷史訓練可能跨 CPU/GPU、版本和重啟，無完整歷史軟硬體證據。",
               "- 也以原評估 seeds 重跑最終模型；是否與 CSV 最後一次評估相符記錄在 data_audit.json 的 logged_final_reproduction_close。若有差異，不能把它當成完全重現。",
               "- elapsed_seconds 會在 resume 歸零，且兩平台不同，不能從最後一列直接報總訓練時間或方法加速比。",
               "- CN win rate 不適用；記錄中的 NaN 並非失敗或零勝率。這份實驗不證明原論文全部結果已復現。", ""]
    if audit["unexpected_configuration_differences"]:
        report += ["**設定有額外差異，結果存在混淆：**", "```json",
                   json.dumps(audit["unexpected_configuration_differences"], ensure_ascii=False, indent=2), "```", ""]
    else:
        report += ["已核對四個 checkpoint 與 resolved_config：未發現允許範圍外的設定差異。完整差異見 data_audit.json。", ""]
    report += ["| Run | 原始列數 | 保留列數 | 回退次數 | 標準評估點數 |", "|---|---:|---:|---:|---:|"]
    for name, info in audit.items():
        if "raw_rows" in info:
            report.append(f"| {name} | {info['raw_rows']} | {info['retained_rows']} | {len(info['rewinds'])} | {info['canonical_eval_rows']} |")
    report += ["", "## Checkpoint 保存建議", "",
               f"總共 {len(manifests)} 個 .pt、{total / 2**30:.3f} GiB。保留各 run 的 best.pt 與 last.pt，其他 step 快照可選擇清理，最多釋放 {optional / 2**30:.3f} GiB。此次未刪除任何檔案。",
               "checkpoint_retention.csv 逐檔列出建議與大小。若要追查剪枝前後或分析訓練中間策略，先額外保留對應 step 檔；CSV 不能還原當時模型。",
               "last.pt：保留完整 actor、critic、optimizer、config 與 step，便於續訓。best.pt：保留驗證最佳模型作輔助分析。",
               "只推論可另存 actor+config+global_step 的精簡副本；可省去 critic/optimizer，但不應直接修改唯一完整 checkpoint。現有 evaluate.py 還會讀 critic，因此精簡副本需要專用載入器。",
               "同一步數的 step_*.pt 和 last.pt 通常重複；此次沒有依檔名就聲稱二進位完全相同。不要因為看似重複就刪除模型內部 GRDC 參數。", "",
               "## 重跑分析", "", "```powershell",
               f".\\.venv\\Scripts\\python.exe compare_experiments.py --output-dir {out.as_posix()} --episodes {args.episodes} --scene-seed {args.scene_seed}",
               "```", "", "analysis_environment.json 保存本次分析版本；所有 CSV 為衍生資料，可在 Excel 開啟。"]
    (out / "REPORT_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "analysis_environment.json").write_text(json.dumps({
        "torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "device": "cpu", "threads": 1, "episodes": args.episodes,
        "scene_seed_start": args.scene_seed, "script_sha256": file_hash(Path(__file__))}, indent=2), encoding="utf-8")
    print(f"Outputs: {out.resolve()}", flush=True)
    print(f"Checkpoint total={total/2**30:.3f} GiB, optional intermediate={optional/2**30:.3f} GiB")


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


if __name__ == "__main__":
    main()

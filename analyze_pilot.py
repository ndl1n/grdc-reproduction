"""Same-budget legacy/v2 CN pilot analysis without modifying source runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_experiments import clean_history, config_diff, evaluate, load_model, save_figure
from grdc.config import load_config


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot_1m_comparison"))
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--scene-seed", type=int, default=910000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = []
    for version in ["legacy", "v2"]:
        for method in ["GRDC", "MAPPO"]:
            suffix = "_v2" if version == "v2" else ""
            name = f"{method.lower()}_cooperative_navigation{suffix}_seed1"
            parent = Path("revised") if version == "v2" else Path()
            definitions.append((version, method, Path("runs") / parent / name,
                                Path("checkpoints") / parent / name))
    final_steps = []
    for version, _, folder, _ in definitions:
        if version == "v2":
            history, _ = clean_history(folder / "metrics.csv")
            final_steps.append(int(history[-1]["step"]))
    if len(set(final_steps)) != 1:
        raise ValueError(f"V2 runs must reach the same budget: {final_steps}")
    budget = final_steps[0]
    all_rows, episode_rows, role_rows, sources, audits, curves = [], [], [], [], {}, {}
    configs = {}
    for version, method, folder, checkpoint_folder in definitions:
        tag = f"{version}_{method}"
        cfg = load_config(folder / "resolved_config.json")
        configs[tag] = cfg.to_dict()
        raw, audit = clean_history(folder / "metrics.csv")
        history = [r for r in raw if int(r["step"]) <= budget]
        interval, stride = cfg.training.eval_interval_steps, cfg.training.rollout_steps
        expected = {((step + stride - 1)//stride)*stride for step in range(interval, budget+1, interval)}
        evaluations = [r for r in history if int(r["step"]) in expected and r["eval_return_mean"]]
        x = np.array([int(r["step"]) for r in evaluations])
        y = np.array([float(r["eval_return_mean"])/float(r["eval_episode_length"]) for r in evaluations])
        if x[-1] != budget or len(x) != len(expected):
            raise ValueError(f"Missing matched evaluation points: {tag}")
        curves[tag] = (x, y, history)
        checkpoint = checkpoint_folder / ("last.pt" if version == "v2" else f"step_{budget}.pt")
        actor, checkpoint_cfg, payload = load_model(checkpoint)
        if int(payload["global_step"]) != budget:
            raise ValueError(f"Wrong checkpoint budget: {checkpoint}")
        if cfg.environment != checkpoint_cfg.environment or cfg.model != checkpoint_cfg.model:
            raise ValueError(f"Checkpoint configuration disagreement: {tag}")
        reproduced, _ = evaluate(actor, checkpoint_cfg, budget, cfg.training.eval_episodes, cfg.seed+100000)
        logged = float(evaluations[-1]["eval_return_mean"])
        difference = np.mean([r["episode_return"] for r in reproduced]) - logged
        if version == "v2" and abs(difference) > 1e-4:
            raise ValueError(f"Logged final evaluation not reproduced: {tag}: {difference}")
        episodes, counts = evaluate(actor, checkpoint_cfg, budget, args.episodes, args.scene_seed)
        for row in episodes:
            episode_rows.append({"version": version, "method": method, "training_seed": cfg.seed, **row})
        if cfg.model.use_roles:
            for role, count in enumerate(counts):
                role_rows.append({"version": version, "role": role, "active": bool(actor.active_prototypes[role]),
                                  "fraction": float(count/counts.sum())})
        finite_keys = ["action_entropy", "policy_loss", "value_loss", "approx_kl", "clip_fraction"]
        if version == "v2":
            finite_keys += ["actor_grad_norm", "actor_grad_norm_max", "actor_grad_clip_scale", "policy_head_grad_norm"]
        bad = [(int(r["step"]), key) for r in history for key in finite_keys if not np.isfinite(float(r[key]))]
        if bad:
            raise ValueError(f"Nonfinite optimization metrics: {tag}: {bad[:3]}")
        heldout = np.array([r["mean_step_reward"] for r in episodes])
        record = {
            "version": version, "method": method, "training_seed": cfg.seed, "step": budget,
            "logged_final_mean_step_reward": float(y[-1]),
            "logged_last5_evaluations_mean": float(y[-5:].mean()),
            "logged_best_eval_reward": float(y.max()),
            "new_scenes_mean_step_reward": float(heldout.mean()),
            "new_scenes_episode_std": float(heldout.std(ddof=1)),
            "collision_pairs_per_step": float(np.mean([r["collision_pairs_per_step"] for r in episodes])),
            "landmark_distance_sum_per_step": float(np.mean([r["sum_landmark_nearest_distance_per_step"] for r in episodes])),
            "active_prototypes": actor.n_active_prototypes if cfg.model.use_roles else "N/A",
            "largest_role_fraction": float(counts.max()/counts.sum()) if cfg.model.use_roles else "N/A",
            "final_action_entropy": float(history[-1]["action_entropy"]),
            "final_role_entropy": float(history[-1]["role_entropy"]) if cfg.model.use_roles else "N/A",
            "max_actor_grad_norm": max(float(r["actor_grad_norm_max"]) for r in history) if version == "v2" else "unlogged",
            "updates_with_clipped_actor_gradients": sum(float(r["actor_grad_clip_scale"]) < 1 for r in history) if version == "v2" else "unlogged",
        }
        all_rows.append(record)
        audit.update(budget=budget, final_reproduction_difference=float(difference),
                     final_reproduction_close=bool(abs(difference) <= 1e-4), finite_optimization_metrics=True)
        audits[tag] = audit
        for path in [folder / "metrics.csv", folder / "resolved_config.json", checkpoint]:
            sources.append({"run": tag, "path": str(path.resolve()), "sha256": digest(path)})
        print(json.dumps(record), flush=True)
    diff = config_diff(configs["v2_GRDC"], configs["v2_MAPPO"])
    allowed = {"experiment_name", "model.use_roles", "model.use_local_graph", "model.use_intra_role_communication",
               "training.role_entropy_coef", "training.prototype_decorrelation_coef", "training.enable_pruning"}
    if any(d["field"] not in allowed for d in diff):
        raise ValueError(f"Unexpected v2 method configuration differences: {diff}")
    audits["v2_method_configuration_differences"] = diff
    audits["new_scenes"] = {"start": args.scene_seed, "episodes": args.episodes,
                            "purpose": "additional diagnostic validation; not untouched final test after subsequent tuning"}
    write_csv(args.output_dir / "summary.csv", all_rows)
    write_csv(args.output_dir / "evaluation_episodes.csv", episode_rows)
    write_csv(args.output_dir / "role_usage.csv", role_rows)
    write_csv(args.output_dir / "sources.csv", sources)
    (args.output_dir / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    colors = {"GRDC": "#2563a6", "MAPPO": "#d66a20"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for version in ["legacy", "v2"]:
        for method in ["GRDC", "MAPPO"]:
            x, y, _ = curves[f"{version}_{method}"]
            axes[0].plot(x/1e6, y, color=colors[method], linestyle="--" if version == "legacy" else "-",
                         marker="o", markersize=3, alpha=.6 if version == "legacy" else 1,
                         label=f"{method} {version}")
    axes[0].set(title="Same-budget learning curves (training seed 1)", xlabel="Environment steps (millions)",
                ylabel="Validation mean step reward (higher is better)")
    axes[0].legend(fontsize=8)
    for index, row in enumerate(all_rows):
        axes[1].scatter(index, row["new_scenes_mean_step_reward"], color=colors[row["method"]], s=65,
                        marker="s" if row["version"] == "v2" else "o")
        axes[1].annotate(f"{row['new_scenes_mean_step_reward']:.3f}", (index,row["new_scenes_mean_step_reward"]),
                         xytext=(0,9),textcoords="offset points",ha="center")
    axes[1].set_xticks(range(4), [f"{r['method']}\n{r['version']}" for r in all_rows])
    axes[1].set(title=f"Same {args.episodes} new scenes at {budget:,} steps", ylabel="Mean step reward", xlabel="One trained model per point; no across-seed CI")
    axes[1].margins(y=.18,x=.15)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
    save_figure(fig,args.output_dir,"pilot_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for method in ["GRDC", "MAPPO"]:
        history = curves[f"v2_{method}"][2]
        axes[0].plot([int(r["step"])/1e6 for r in history], [float(r["actor_grad_norm_max"]) for r in history],
                     color=colors[method], linewidth=.8, alpha=.8, label=method)
    axes[0].axhline(10,color=".4",linestyle=":",label="Global clipping threshold")
    axes[0].set(title="V2 gradient safety", xlabel="Environment steps (millions)",ylabel="Max actor gradient norm per update")
    for version in ["legacy","v2"]:
        history=curves[f"{version}_GRDC"][2]
        axes[1].plot([int(r["step"])/1e6 for r in history],[int(r["active_prototypes"]) for r in history],label=f"GRDC {version}")
    axes[1].set(title="GRDC prototype pruning",xlabel="Environment steps (millions)",ylabel="Active prototypes")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
    save_figure(fig,args.output_dir,"pilot_diagnostics")
    report=["# CN 約 1M pilot 比較", "",f"四個模型皆為 training seed 1、{budget:,} environment steps；新版 horizon 仍為 10M。", "",
            "| 版本 | 方法 | 原評估末點 | 最近5次評估平均 | 共同新場景平均 |", "|---|---|---:|---:|---:|"]
    for r in all_rows:
        report.append(f"| {r['version']} | {r['method']} | {r['logged_final_mean_step_reward']:.6f} | {r['logged_last5_evaluations_mean']:.6f} | {r['new_scenes_mean_step_reward']:.6f} |")
    report += ["", "數值為每步團隊獎勵，越高越好。原評估為32 episodes；新場景為128 episodes，所有模型使用同一組場景。", "",
               "- 原始紀錄未修改；舊版回退分支僅在記憶體清理，主圖只取共同100k門檻，不含續訓補追的密集評估。",
               "- v2 原評估末點由對應 checkpoint 核對；legacy 的回放差值完整保留在 audit.json，差異原因尚未定位，不把差值隱藏或改寫CSV。v2 共同設定只有GRDC模組／角色正則／剪枝不同。",
               "- 舊版與新版同時改了數值安全、公式尺度、動作歷史與critic，不可把所有差異歸因於單獨修正梯度。",
               "- 只有一個 training seed；episode SD 不等於跨training-seed標準差，不宣稱統計顯著或已復現整篇論文。",
               "- 新場景作本次診斷，若依此決定調參，未來正式測試需另留未使用場景。",
               "- prototype 10不表示10個功能角色已被驗證；請併看 deterministic role usage。MAPPO的原型數不適用。", "",
               "資料：summary.csv、evaluation_episodes.csv、role_usage.csv；來源雜湊：sources.csv；稽核：audit.json。"]
    revised_grdc = next(r for r in all_rows if r["version"] == "v2" and r["method"] == "GRDC")
    revised_mappo = next(r for r in all_rows if r["version"] == "v2" and r["method"] == "MAPPO")
    old_grdc = next(r for r in all_rows if r["version"] == "legacy" and r["method"] == "GRDC")
    report += ["", "## 判讀與下一步", "",
               f"共同新場景中，新版 GRDC 相對舊版同預算增加 {revised_grdc['new_scenes_mean_step_reward']-old_grdc['new_scenes_mean_step_reward']:.6f} reward/step。",
               f"新版 GRDC−MAPPO：最近5次原評估平均差 {revised_grdc['logged_last5_evaluations_mean']-revised_mappo['logged_last5_evaluations_mean']:+.6f}；共同新場景差 {revised_grdc['new_scenes_mean_step_reward']-revised_mappo['new_scenes_mean_step_reward']:+.6f}。正值才支持 GRDC。",
               f"新版 GRDC 最高原始梯度範數 {revised_grdc['max_actor_grad_norm']:.6f}，{revised_grdc['updates_with_clipped_actor_gradients']} 個 rollout 更新有裁剪；新場景最大角色比例 {revised_grdc['largest_role_fraction']:.1%}，活躍原型 {revised_grdc['active_prototypes']}。",
               "這支持修正後可正常學習、較舊版改善，但不支持已穩定超越 MAPPO。角色分配分散也不等於已驗證功能分工。",
               "建議先固定參數補 seed 2 的相同 1M pilot，再判斷跨 seed 是否一致；暫不直接投入新的10M或為了讓GRDC勝出而單邊調參。"]
    (args.output_dir/"REPORT_zh.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    if any(digest(Path(s["path"])) != s["sha256"] for s in sources):
        raise RuntimeError("Source changed during analysis")
    print(f"Outputs: {args.output_dir.resolve()}",flush=True)


if __name__=="__main__":
    main()

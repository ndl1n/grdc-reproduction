"""Check the settings that can be checked locally; never infer missing Appendix A values."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from grdc.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/paper_alignment/config_checks.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    records, failures = [], []
    for path in sorted((root / "configs/revised").glob("*.json")):
        cfg = load_config(path)
        checks = {
            "10m_figure_budget": cfg.training.total_steps == 10000000,
            "32_evaluation_episodes": cfg.training.eval_episodes == 32,
            "paper_equation_mode": cfg.model.paper_equations,
            "safe_attention": cfg.model.stable_role_attention,
            "observation_action_history": cfg.model.include_previous_action,
            "one_step_td_critic": cfg.training.critic_objective == "td",
            "isolated_output": cfg.output_dir == "runs/revised" and cfg.checkpoint_dir == "checkpoints/revised",
        }
        if cfg.environment.name != "smacv2":
            checks["vision_radius_0_3"] = cfg.environment.vision_radius == .3
        else:
            checks["paper_scenario_name"] = (
                cfg.environment.race in {"protoss", "terran", "zerg"} and
                (cfg.environment.n_agents, cfg.environment.n_enemies) in {(5, 5), (10, 10), (20, 23)})
        records.append({"config": str(path.relative_to(root)), "checks": checks,
                        "resolved_config": cfg.to_dict()})
        failures.extend(f"{path.name}: {key}" for key, value in checks.items() if not value)
    if len(records) != 14:
        failures.append(f"Expected 14 revised configs, got {len(records)}")
    pair_checks = []
    for name in ["cooperative_navigation", "predator_prey"]:
        a = load_config(root / f"configs/revised/{name}.json")
        b = load_config(root / f"configs/revised/mappo_{name}.json")
        left, right = a.to_dict()["training"], b.to_dict()["training"]
        for field in ["role_entropy_coef", "prototype_decorrelation_coef", "enable_pruning"]:
            left.pop(field)
            right.pop(field)
        checks = {"same_environment": a.environment == b.environment,
                  "same_common_learning_settings": left == right,
                  "same_history": a.model.include_previous_action == b.model.include_previous_action,
                  "same_hidden_size": a.model.hidden_dim == b.model.hidden_dim,
                  "baseline_modules_disabled": not any([b.model.use_roles, b.model.use_local_graph,
                                                        b.model.use_intra_role_communication])}
        pair_checks.append({"task": name, "checks": checks})
        failures.extend(f"{name}: {key}" for key, value in checks.items() if not value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "scope": "configuration checks only; not proof of author environment/hyperparameter equivalence",
        "paper_source": "Gao et al. AAAI-26, supplied 9-page PDF, pp. 29443-29447",
        "official_page": "https://ojs.aaai.org/index.php/AAAI/article/view/40185",
        "configs": records, "method_pairs": pair_checks, "failures": failures,
        "unresolved": ["Missing Appendix A", "Exact CN/PP environment and reward dynamics",
                       "K=10 inferred from Figure 7, not a universal explicit setting",
                       "Hidden sizes, optimizer/PPO/regularization/pruning numeric settings",
                       "Target critic synchronization, role-gradient estimator and parameter sharing",
                       "Full four-seed experiments and all comparison methods",
                       "SMACv2/StarCraft II installation, capabilities and runtime validation"],
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [root / "grdc/models.py", root / "grdc/trainer.py", root / "grdc/config.py"]}}
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Checked {len(records)} configs; {len(failures)} failures. Output: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

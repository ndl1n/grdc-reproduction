"""Generate explicit, isolated v2 configs from the existing experiment settings.

All numerical hyperparameters absent from the paper remain assumptions.
This generator never rewrites the original configs or an edited v2 config.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def revised(source):
    config = copy.deepcopy(source)
    config["experiment_name"] += "_v2"
    config["output_dir"] = "runs/revised"
    config["checkpoint_dir"] = "checkpoints/revised"
    config["model"].update(paper_equations=True, include_previous_action=True,
                           stable_role_attention=True, min_prototypes=1)
    config["training"].update(critic_objective="td", keep_step_checkpoints=False)
    return config


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "configs" / "revised"
    output.mkdir(parents=True, exist_ok=True)
    generated = {}
    for path in sorted((root / "configs").glob("*.json")):
        if path.name.startswith("mappo_"):
            continue
        config = revised(json.loads(path.read_text(encoding="utf-8")))
        generated[path.name] = config
        if path.name in {"cooperative_navigation.json", "predator_prey.json"}:
            baseline = copy.deepcopy(config)
            baseline["experiment_name"] = baseline["experiment_name"].replace("grdc_", "mappo_", 1)
            baseline["model"].update(use_roles=False, use_local_graph=False, use_intra_role_communication=False)
            baseline["training"].update(role_entropy_coef=0.0, prototype_decorrelation_coef=0.0, enable_pruning=False)
            generated["mappo_" + path.name] = baseline
    no_pruning = copy.deepcopy(generated["cooperative_navigation.json"])
    no_pruning["experiment_name"] += "_no_pruning"
    no_pruning["training"]["enable_pruning"] = False
    generated["cooperative_navigation_no_pruning.json"] = no_pruning
    for name, config in generated.items():
        path = output / name
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != config:
                raise RuntimeError(f"Refusing to overwrite edited config: {path}")
            continue
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(generated)} configs in {output}")


if __name__ == "__main__":
    main()

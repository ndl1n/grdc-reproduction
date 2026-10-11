from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export role prototypes and their 3D PCA projection")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="role_analysis")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["actor"]
    prototypes = state["prototypes"].detach().cpu().numpy()
    active = state["active_prototypes"].detach().cpu().numpy().astype(bool)
    centered = prototypes - prototypes.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[: min(3, right.shape[0])]
    projection = centered @ components.T
    if projection.shape[1] < 3:
        projection = np.pad(projection, ((0, 0), (0, 3 - projection.shape[1])))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prototypes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "active", *[f"dim_{i}" for i in range(prototypes.shape[1])]])
        for role, vector in enumerate(prototypes):
            writer.writerow([role, int(active[role]), *vector.tolist()])
    with (output_dir / "prototype_pca_3d.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "active", "pc1", "pc2", "pc3"])
        for role, point in enumerate(projection):
            writer.writerow([role, int(active[role]), *point.tolist()])
    print(f"Wrote role analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()


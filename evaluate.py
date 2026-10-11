from __future__ import annotations

import argparse
import json

import torch

from grdc.config import config_from_dict
from grdc.trainer import GRDCTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic GRDC policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    config.device = args.device
    trainer = GRDCTrainer(config)
    try:
        trainer.load_checkpoint(args.checkpoint, load_optimizers=False)
        print(json.dumps(trainer.evaluate(args.episodes), indent=2, ensure_ascii=False))
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    main()


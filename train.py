from __future__ import annotations

import argparse

from grdc.config import load_config
from grdc.trainer import GRDCTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRDC with recurrent MAPPO")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument("--resume", help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, help="Override the config seed")
    parser.add_argument("--device", help="Override device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--total-steps", type=int, help="Override total environment steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.total_steps is not None:
        config.training.total_steps = args.total_steps
    config.validate()
    trainer = GRDCTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume, load_optimizers=True)
    trainer.train()


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import signal
from pathlib import Path

import torch

from grdc.config import load_config, config_from_dict
from grdc.trainer import GRDCTrainer, validate_checkpoint_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRDC with recurrent MAPPO")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument("--resume", help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, help="Override the config seed")
    parser.add_argument("--device", help="Override device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--total-steps", type=int, help="Override total environment steps")
    parser.add_argument("--stop-after-steps", type=int,
                        help="Stop at this absolute step without changing the configured annealing horizon")
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
    if config.model.use_roles and not config.model.stable_role_attention:
        raise ValueError("Legacy role attention is for old-model evaluation only. Start a fresh run with configs/revised/")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("--stop-after-steps must be positive")
    metrics_path = Path(config.output_dir) / f"{config.experiment_name}_seed{config.seed}" / "metrics.csv"
    if not args.resume and metrics_path.exists() and metrics_path.stat().st_size:
        raise ValueError(f"Run already contains metrics: {metrics_path}. Use --resume or a new experiment_name")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint_config(config, config_from_dict(saved["config"]))
    trainer = GRDCTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume, load_optimizers=True)
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        if trainer.stop_requested:
            raise KeyboardInterrupt("Forced stop; use the last complete checkpoint")
        trainer.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    try:
        trainer.train(stop_after_steps=args.stop_after_steps)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()


from dataclasses import replace
import copy
import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grdc.config import load_config, TrainingConfig
from grdc.trainer import GRDCTrainer, critic_loss


@pytest.fixture
def trainer(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config("configs/revised/cooperative_navigation.json")
    config.output_dir = str(tmp_path / "runs")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.device = "cpu"
    config.environment.n_agents = config.environment.n_targets = 3
    config.environment.episode_limit = 8
    config.model.hidden_dim, config.model.attention_dim = 16, 8
    config.model.n_prototypes = 4
    config.training.total_steps = 1000
    config.training.rollout_steps = 32
    config.training.recurrent_chunk_length = 8
    config.training.minibatch_chunks = 2
    config.training.update_epochs = 2
    config.training.eval_episodes = 2
    config.training.eval_interval_steps = config.training.save_interval_steps = 64
    instance = GRDCTrainer(config)
    try:
        yield instance
    finally:
        instance.env.close()
        if instance.writer is not None:
            instance.writer.close()
        torch.set_num_threads(previous_threads)


def test_previous_actions_reset_at_episode_boundaries_and_replay_probabilities(trainer):
    trainer.collect_rollout()
    buffer = trainer.buffer
    for i in range(buffer.position):
        if i % 8 == 0:
            assert np.all(buffer.previous_actions[i] == -1)
        else:
            np.testing.assert_array_equal(buffer.previous_actions[i], buffer.actions[i-1])
        output = trainer.actor.forward_step(
            trainer._tensor(buffer.observations[i])[None], trainer._tensor(buffer.hidden[i])[None],
            trainer._tensor(buffer.adjacency[i])[None], trainer._tensor(buffer.available_actions[i])[None],
            trainer.temperature(), forced_roles=trainer._tensor(buffer.role_assignments[i], torch.long)[None],
            previous_actions=trainer._tensor(buffer.previous_actions[i], torch.long)[None])
        logp = torch.distributions.Categorical(logits=output.logits).log_prob(
            trainer._tensor(buffer.actions[i], torch.long)[None])
        torch.testing.assert_close(logp.detach().squeeze(0), trainer._tensor(buffer.log_probabilities[i]),
                                   atol=1e-6, rtol=1e-5)
    before = trainer.actor.policy[-1].weight.detach().clone()
    metrics = trainer.update()
    assert np.isfinite(metrics.actor_grad_norm_max) and metrics.actor_grad_norm_max < 1000
    assert metrics.policy_head_grad_norm > 0
    assert not torch.equal(before, trainer.actor.policy[-1].weight)
    evaluation = trainer.evaluate(2)
    assert evaluation["eval_mean_step_reward"] == pytest.approx(evaluation["eval_return_mean"] / 8)


def test_checkpoint_roundtrip_restores_optimizer_and_rng(trainer):
    trainer.collect_rollout()
    trainer.update()
    checkpoint = trainer.save_checkpoint("roundtrip.pt")
    expected = (random.random(), np.random.random(), torch.rand(3))
    trainer.collect_rollout()
    trainer.load_checkpoint(checkpoint)
    assert trainer.global_step == 32 and trainer.update_number == 1
    assert trainer.actor_optimizer.state
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
    assert not checkpoint.with_name(checkpoint.name + ".tmp").exists()


def test_resume_rejects_silent_model_or_learning_setting_changes(trainer):
    checkpoint = trainer.save_checkpoint("unchanged.pt")
    trainer.config.training.total_steps = 2000
    with pytest.raises(ValueError, match="learning settings"):
        trainer.load_checkpoint(checkpoint)
    trainer.config.training.total_steps = 1000
    trainer.config.model.paper_equations = False
    with pytest.raises(ValueError, match="model/environment"):
        trainer.load_checkpoint(checkpoint)


def test_resume_schedule_skips_past_intervals_and_does_not_grow_step_files(trainer):
    trainer.global_step = 160
    trainer.update_number = 5
    calls = []

    def evaluation(*args, **kwargs):
        calls.append(trainer.global_step)
        return {"eval_return_mean": 0., "eval_return_std": 0., "eval_episode_length": 8.}

    trainer.evaluate = evaluation
    trainer.train(stop_after_steps=224)
    assert calls == [192]
    assert not list(trainer.checkpoint_dir.glob("step_*.pt"))
    assert (trainer.checkpoint_dir / "latest.pt").exists()
    assert trainer.config.training.total_steps == 1000
    assert trainer.temperature() > .7


def test_graceful_stop_saves_only_after_complete_update(trainer):
    trainer.request_stop()
    trainer.train()
    payload = torch.load(trainer.checkpoint_dir / "interrupt.pt", weights_only=False)
    assert payload["global_step"] == 32 and payload["update_number"] == 1
    assert payload["rng_state"] and payload["actor_optimizer"]["state"]
    assert (trainer.checkpoint_dir / "last.pt").exists()


def test_td_critic_uses_one_step_target_instead_of_gae_returns():
    values = torch.tensor([0., 2.], requires_grad=True)
    old = torch.zeros(2)
    returns = torch.full((2,), 100.)
    td_targets = torch.tensor([1., 3.])
    cfg = TrainingConfig(critic_objective="td")
    torch.testing.assert_close(critic_loss(values, old, returns, td_targets, cfg), torch.tensor(1.))
    assert critic_loss(values, old, returns, td_targets, replace(cfg, critic_objective="ppo_clipped")) > 1000


def test_evaluation_does_not_touch_existing_run_files(trainer):
    checkpoint = trainer.save_checkpoint("evaluation.pt")
    if trainer.writer is not None:
        trainer.writer.flush()
    before = {p: p.read_bytes() for p in trainer.run_dir.rglob("*") if p.is_file()}
    config = copy.deepcopy(trainer.config)
    config.device = "auto"
    evaluation = GRDCTrainer(config, record_run=False)
    try:
        evaluation.load_checkpoint(checkpoint, load_optimizers=False)
        assert np.isfinite(evaluation.evaluate(2)["eval_return_mean"])
        with pytest.raises(RuntimeError, match="cannot write"):
            evaluation.save_checkpoint("not_allowed.pt")
    finally:
        evaluation.env.close()
    after = {p: p.read_bytes() for p in trainer.run_dir.rglob("*") if p.is_file()}
    assert before == after


def test_evaluation_does_not_create_directories(trainer, tmp_path):
    config = copy.deepcopy(trainer.config)
    config.output_dir = str(tmp_path / "must_not_create_runs")
    config.checkpoint_dir = str(tmp_path / "must_not_create_checkpoints")
    evaluation = GRDCTrainer(config, record_run=False)
    try:
        evaluation.evaluate(1)
        assert not evaluation.run_dir.exists()
        assert not evaluation.checkpoint_dir.exists()
        assert evaluation.writer is None and evaluation.csv is None
    finally:
        evaluation.env.close()


def test_existing_config_is_not_overwritten_on_conflicting_run(trainer):
    path = trainer.run_dir / "resolved_config.json"
    before = path.read_bytes()
    config = copy.deepcopy(trainer.config)
    config.training.actor_lr *= 2
    with pytest.raises(ValueError, match="learning settings"):
        GRDCTrainer(config)
    assert path.read_bytes() == before


def test_cli_checks_wrong_resume_seed_before_creating_run(trainer, tmp_path, monkeypatch):
    import argparse
    import train
    checkpoint = trainer.save_checkpoint("preflight.pt")
    config = copy.deepcopy(trainer.config)
    config.output_dir = str(tmp_path / "new_runs")
    config.checkpoint_dir = str(tmp_path / "new_checkpoints")
    path = tmp_path / "resume_config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    args = argparse.Namespace(config=str(path), resume=str(checkpoint), seed=2, device="cpu",
                              total_steps=None, stop_after_steps=64)
    monkeypatch.setattr(train, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="Resume seed"):
        train.main()
    assert not (tmp_path / "new_runs").exists()
    assert not (tmp_path / "new_checkpoints").exists()


def test_resumed_next_update_matches_uninterrupted_cpu_update(trainer):
    trainer.collect_rollout()
    trainer.update()
    checkpoint = trainer.save_checkpoint("continuation.pt")
    trainer.collect_rollout()
    first_metrics = trainer.update()
    expected_actor = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
    expected_critic = {k: v.clone() for k, v in trainer.critic.state_dict().items()}
    trainer.load_checkpoint(checkpoint)
    trainer.collect_rollout()
    resumed_metrics = trainer.update()
    for key, value in expected_actor.items():
        torch.testing.assert_close(trainer.actor.state_dict()[key], value, atol=0, rtol=0)
    for key, value in expected_critic.items():
        torch.testing.assert_close(trainer.critic.state_dict()[key], value, atol=0, rtol=0)
    assert first_metrics == resumed_metrics

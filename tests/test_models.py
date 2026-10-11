from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from grdc.config import ModelConfig
from grdc.models import GRDCActor


def test_actor_shapes_masks_and_pruning() -> None:
    config = ModelConfig(hidden_dim=32, attention_dim=16, n_prototypes=6, min_prototypes=2)
    actor = GRDCActor(obs_dim=11, n_actions=5, config=config)
    batch, agents = 3, 4
    observations = torch.randn(batch, agents, 11)
    hidden = actor.initial_hidden(batch, agents, torch.device("cpu"))
    adjacency = torch.ones(batch, agents, agents, dtype=torch.bool)
    adjacency[:, torch.arange(agents), torch.arange(agents)] = False
    available = torch.ones(batch, agents, 5, dtype=torch.bool)
    available[:, :, 4] = False
    output = actor.forward_step(
        observations, hidden, adjacency, available, temperature=0.7
    )
    assert output.logits.shape == (batch, agents, 5)
    assert output.hidden.shape == (batch, agents, 32)
    assert output.role_probabilities.shape == (batch, agents, 6)
    assert torch.all(output.logits[:, :, 4] < -1e8)
    utilization = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    pruned = actor.prune(utilization, sensitivity=0.11)
    assert len(pruned) == 4
    assert actor.n_active_prototypes == 2


def test_no_neighbour_graph_is_finite() -> None:
    actor = GRDCActor(7, 3, ModelConfig(hidden_dim=16, attention_dim=8))
    output = actor.forward_step(
        torch.randn(2, 3, 7),
        actor.initial_hidden(2, 3, torch.device("cpu")),
        torch.zeros(2, 3, 3, dtype=torch.bool),
        torch.ones(2, 3, 3, dtype=torch.bool),
        temperature=1.0,
    )
    assert torch.isfinite(output.logits).all()

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


@pytest.mark.parametrize("paper", [False, True])
def test_singleton_roles_have_zero_message_and_zero_backward_path(paper):
    torch.manual_seed(1)
    actor = GRDCActor(7, 3, ModelConfig(hidden_dim=16, attention_dim=8, paper_equations=paper))
    hidden = torch.randn(1, 3, 16, requires_grad=True)
    roles = torch.nn.functional.one_hot(torch.tensor([[0, 1, 2]]), 10).float().requires_grad_()
    message = actor._intra_role_message(hidden, roles, torch.ones(1, 3, dtype=torch.bool))
    assert torch.equal(message, torch.zeros_like(message))
    message.sum().backward()
    assert torch.equal(roles.grad, torch.zeros_like(roles))
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("roles", [[0, 1, 2, 3, 4], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0]])
def test_role_routing_backward_does_not_suppress_policy_gradient(paper, roles):
    torch.manual_seed(1)
    actor = GRDCActor(39, 5, ModelConfig(paper_equations=paper))
    output = actor.forward_step(
        torch.randn(1, 5, 39), actor.initial_hidden(1, 5, torch.device("cpu")),
        torch.ones(1, 5, 5, dtype=torch.bool), torch.ones(1, 5, 5, dtype=torch.bool),
        temperature=1.0, forced_roles=torch.tensor([roles]))
    (-output.logits.log_softmax(-1)[0, :, 0].mean()).backward()
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0, error_if_nonfinite=True)
    assert norm < 100
    assert actor.policy[-1].weight.grad.norm() > .01


@pytest.mark.parametrize("paper", [False, True])
def test_unrelated_large_attention_score_cannot_erase_valid_message(paper):
    actor = GRDCActor(2, 2, ModelConfig(hidden_dim=2, attention_dim=2, paper_equations=paper))
    with torch.no_grad():
        for layer in [actor.role_query, actor.role_key, actor.role_value]:
            layer.weight.copy_(torch.eye(2))
    hidden = torch.tensor([[[1., 0.], [0., 1.], [1000., 0.]]])
    roles = torch.nn.functional.one_hot(torch.tensor([[0, 0, 1]]), 10).float()
    message = actor._intra_role_message(hidden, roles, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(message, torch.tensor([[[0., 1.], [1., 0.], [0., 0.]]]))


def test_paper_equations_match_unscaled_dot_product_and_frobenius_sum():
    actor = GRDCActor(4, 3, ModelConfig(hidden_dim=8, attention_dim=4, n_prototypes=3,
                                       paper_equations=True))
    hidden = torch.randn(1, 3, 8)
    adjacency = ~torch.eye(3, dtype=torch.bool)[None]
    scores = actor.graph_query(hidden) @ actor.graph_key(hidden).transpose(-1, -2)
    expected = scores.masked_fill(~adjacency, -torch.inf).softmax(-1) @ actor.graph_value(hidden)
    torch.testing.assert_close(actor._graph_context(hidden, adjacency), expected)
    domain = torch.randn(1, 3, 16)
    probabilities, _, _ = actor._role_assignment(domain, .5, True, None)
    torch.testing.assert_close(probabilities, (domain @ actor.prototypes.T / .5).softmax(-1))
    with torch.no_grad():
        actor.prototypes.fill_(1)
    torch.testing.assert_close(actor.prototype_decorrelation_loss(), torch.tensor(6.))


def test_paper_gumbel_temperature_does_not_change_hard_sample_with_same_noise():
    actor = GRDCActor(4, 3, ModelConfig(hidden_dim=8, attention_dim=4, paper_equations=True))
    domain = torch.randn(2, 5, 16)
    torch.manual_seed(7)
    _, first, _ = actor._role_assignment(domain, 1., False, None)
    torch.manual_seed(7)
    _, second, _ = actor._role_assignment(domain, .1, False, None)
    assert torch.equal(first, second)


def test_legacy_message_preserves_original_operation_order():
    import math
    torch.manual_seed(9)
    actor = GRDCActor(7, 3, ModelConfig(hidden_dim=16, attention_dim=8, stable_role_attention=False))
    hidden = torch.randn(1, 4, 16)
    vectors = torch.nn.functional.one_hot(torch.tensor([[0, 0, 1, 2]]), 10).float()
    alive = torch.ones(1, 4, dtype=torch.bool)
    scores = torch.einsum("bnd,bmd->bnm", actor.role_query(hidden), actor.role_key(hidden)) / math.sqrt(8)
    scores = scores - scores.amax(-1, keepdim=True).detach()
    affinity = torch.einsum("bnk,bmk->bnm", vectors, vectors)
    peer = ~torch.eye(4, dtype=torch.bool)[None]
    numerator = torch.exp(scores) * affinity.clamp_min(0) * peer.float()
    weights = numerator / numerator.sum(-1, keepdim=True).clamp_min(1e-12)
    expected = torch.einsum("bnm,bmh->bnh", weights, actor.role_value(hidden))
    torch.testing.assert_close(actor._intra_role_message(hidden, vectors, alive), expected, atol=0, rtol=0)

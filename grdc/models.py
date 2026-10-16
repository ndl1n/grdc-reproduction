from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from grdc.config import ModelConfig


def _masked_attention(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Stable softmax that returns all zeros for rows with no valid keys."""
    mask = mask.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_scores, dim=-1) * mask.to(scores.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _init_module(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass
class ActorOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    role_probabilities: torch.Tensor
    role_assignments: torch.Tensor
    role_vectors: torch.Tensor


class GRDCActor(nn.Module):
    """Parameter-shared GRDC actor for homogeneous cooperative agents."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = config.hidden_dim
        self.attention_dim = config.attention_dim
        self.max_prototypes = config.n_prototypes
        self.min_prototypes = config.min_prototypes
        self.use_roles = config.use_roles
        self.use_local_graph = config.use_local_graph
        self.use_intra_role_communication = config.use_intra_role_communication
        self.paper_equations = config.paper_equations
        self.include_previous_action = config.include_previous_action
        self.stable_role_attention = config.stable_role_attention

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim + (n_actions if self.include_previous_action else 0), config.hidden_dim), nn.Tanh(),
        )
        self.trajectory_encoder = nn.GRUCell(config.hidden_dim, config.hidden_dim)

        self.graph_query = nn.Linear(config.hidden_dim, config.attention_dim, bias=False)
        self.graph_key = nn.Linear(config.hidden_dim, config.attention_dim, bias=False)
        self.graph_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        domain_dim = 2 * config.hidden_dim
        self.prototypes = nn.Parameter(torch.empty(config.n_prototypes, domain_dim))
        nn.init.orthogonal_(self.prototypes)
        self.register_buffer(
            "active_prototypes", torch.ones(config.n_prototypes, dtype=torch.bool)
        )

        self.role_query = nn.Linear(config.hidden_dim, config.attention_dim, bias=False)
        self.role_key = nn.Linear(config.hidden_dim, config.attention_dim, bias=False)
        self.role_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.prototype_to_message = nn.Linear(domain_dim, config.hidden_dim)

        self.policy = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, n_actions),
        )
        if config.orthogonal_init:
            self.apply(_init_module)
            nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    @property
    def n_active_prototypes(self) -> int:
        return int(self.active_prototypes.sum().item())

    def initial_hidden(self, batch_size: int, n_agents: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, n_agents, self.hidden_dim, device=device)

    def _graph_context(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        query = self.graph_query(hidden)
        key = self.graph_key(hidden)
        value = self.graph_value(hidden)
        scores = torch.einsum("bnd,bmd->bnm", query, key)
        if not self.paper_equations:
            scores = scores / math.sqrt(self.attention_dim)
        weights = _masked_attention(scores, adjacency)
        return torch.einsum("bnm,bmh->bnh", weights, value)

    def _role_assignment(
        self,
        domain: torch.Tensor,
        temperature: float,
        deterministic: bool,
        forced_roles: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.einsum("bnd,kd->bnk", domain, self.prototypes)
        if not self.paper_equations:
            logits = logits / math.sqrt(domain.shape[-1])
        logits = logits.masked_fill(~self.active_prototypes.view(1, 1, -1), -1e9)
        tau = max(float(temperature), 1e-4)
        probabilities = torch.softmax(logits / tau, dim=-1)

        if forced_roles is not None:
            hard = F.one_hot(forced_roles.long(), self.max_prototypes).to(domain.dtype)
            role_vectors = (hard + (probabilities - probabilities.detach()) if self.stable_role_attention
                            else hard + probabilities - probabilities.detach())
            assignments = forced_roles.long()
        elif deterministic:
            assignments = logits.argmax(dim=-1)
            hard = F.one_hot(assignments, self.max_prototypes).to(domain.dtype)
            role_vectors = (hard + (probabilities - probabilities.detach()) if self.stable_role_attention
                            else hard + probabilities - probabilities.detach())
        else:
            # Standard Gumbel-Softmax temperature controls relaxation, not the
            # hard categorical sampling distribution. Keep old behavior for old configs.
            role_vectors = F.gumbel_softmax(
                logits if self.paper_equations else logits / tau,
                tau=tau if self.paper_equations else 1.0, hard=True, dim=-1)
            assignments = role_vectors.argmax(dim=-1)
        return probabilities, assignments, role_vectors

    def _intra_role_message(
        self,
        hidden: torch.Tensor,
        role_vectors: torch.Tensor,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        query = self.role_query(hidden)
        key = self.role_key(hidden)
        # Eq. (3) aggregates h_tau itself, rather than a learned value projection.
        value = hidden if self.paper_equations else self.role_value(hidden)
        scores = torch.einsum("bnd,bmd->bnm", query, key)
        if not self.paper_equations:
            scores = scores / math.sqrt(self.attention_dim)

        if not self.stable_role_attention:
            # Retrospective evaluation only. Preserve the original operation
            # order; training this path is rejected by the CLI.
            scores = scores - scores.amax(dim=-1, keepdim=True).detach()
            compatibility = torch.einsum("bnk,bmk->bnm", role_vectors, role_vectors)
            not_self = ~torch.eye(hidden.shape[1], dtype=torch.bool, device=hidden.device)[None]
            mask = not_self & alive.unsqueeze(1) & alive.unsqueeze(2)
            numerator = scores.exp() * compatibility.clamp_min(0) * mask.to(scores.dtype)
            weights = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return torch.einsum("bnm,bmh->bnh", weights, value)

        # Straight-through role vectors are exactly one-hot in the forward pass,
        # but keep gradients from the policy path to the role matcher.
        compatibility = torch.einsum("bnk,bmk->bnm", role_vectors, role_vectors)
        n_agents = hidden.shape[1]
        not_self = ~torch.eye(n_agents, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        peer_mask = not_self & alive.unsqueeze(1) & alive.unsqueeze(2)
        assignments = role_vectors.detach().argmax(dim=-1)
        same_role = assignments.unsqueeze(2) == assignments.unsqueeze(1)
        eligible = peer_mask & same_role
        has_peer = eligible.any(dim=-1, keepdim=True)
        # Center only on eligible peers: an unrelated large score must not
        # underflow all valid weights. Empty groups use a constant denominator,
        # and a zero backward path, rather than differentiating through 1e-12.
        maximum = scores.masked_fill(~eligible, -torch.inf).amax(dim=-1, keepdim=True)
        maximum = torch.where(has_peer, maximum, torch.zeros_like(maximum)).detach()
        # Off-role edges have zero forward weight. Bounding their exponential
        # also bounds the straight-through surrogate derivative at zero affinity.
        exponentials = torch.exp((scores - maximum).clamp(max=0.0))
        unnormalized = exponentials * compatibility.clamp_min(0.0) * peer_mask.to(scores.dtype)
        denominator = torch.where(has_peer, unnormalized.sum(dim=-1, keepdim=True),
                                  torch.ones_like(maximum))
        weights = (unnormalized / denominator) * has_peer.to(scores.dtype)
        return torch.einsum("bnm,bmh->bnh", weights, value)

    def forward_step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        available_actions: torch.Tensor,
        temperature: float,
        deterministic: bool = False,
        forced_roles: torch.Tensor | None = None,
        previous_actions: torch.Tensor | None = None,
    ) -> ActorOutput:
        batch_size, n_agents, _ = observations.shape
        if self.include_previous_action:
            if previous_actions is None:
                previous_onehot = observations.new_zeros(batch_size, n_agents, self.n_actions)
            else:
                if previous_actions.shape != observations.shape[:2]:
                    raise ValueError("previous_actions must have shape [batch, agents]")
                previous_onehot = F.one_hot(previous_actions.clamp_min(0).long(), self.n_actions)
                previous_onehot = previous_onehot.to(observations.dtype) * (previous_actions >= 0).unsqueeze(-1)
            observations = torch.cat([observations, previous_onehot], dim=-1)
        encoded = self.obs_encoder(observations).reshape(batch_size * n_agents, -1)
        next_hidden = self.trajectory_encoder(
            encoded, hidden.reshape(batch_size * n_agents, -1)
        ).reshape(batch_size, n_agents, -1)

        graph_context = (
            self._graph_context(next_hidden, adjacency.bool())
            if self.use_local_graph
            else torch.zeros_like(next_hidden)
        )
        domain = torch.cat([next_hidden, graph_context], dim=-1)
        if self.use_roles:
            role_probabilities, role_assignments, role_vectors = self._role_assignment(
                domain, temperature, deterministic, forced_roles
            )
        else:
            role_assignments = torch.zeros(
                batch_size, n_agents, dtype=torch.long, device=observations.device
            )
            role_vectors = F.one_hot(
                role_assignments, self.max_prototypes
            ).to(observations.dtype)
            role_probabilities = role_vectors
        alive = available_actions.bool().sum(dim=-1) > 1
        # Particle tasks have all actions available; dead SMACv2 units only have no-op.
        alive = alive | (available_actions.bool().sum(dim=-1) == self.n_actions)
        if self.use_intra_role_communication and self.use_roles:
            message = self._intra_role_message(next_hidden, role_vectors, alive)
        elif self.use_roles:
            selected_prototype = torch.einsum(
                "bnk,kd->bnd", role_vectors, self.prototypes
            )
            message = torch.tanh(self.prototype_to_message(selected_prototype))
        else:
            message = torch.zeros_like(next_hidden)
        logits = self.policy(torch.cat([next_hidden, message], dim=-1))
        logits = logits.masked_fill(~available_actions.bool(), -1e9)
        return ActorOutput(
            logits=logits,
            hidden=next_hidden,
            role_probabilities=role_probabilities,
            role_assignments=role_assignments,
            role_vectors=role_vectors,
        )

    def role_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        marginal = probabilities.reshape(-1, self.max_prototypes).mean(dim=0)
        active = self.active_prototypes
        marginal = marginal[active]
        marginal = marginal / marginal.sum().clamp_min(1e-12)
        return -(marginal * marginal.clamp_min(1e-12).log()).sum()

    def prototype_decorrelation_loss(self) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes[self.active_prototypes], dim=-1)
        cosine = prototypes @ prototypes.transpose(0, 1)
        identity = torch.eye(cosine.shape[0], device=cosine.device, dtype=cosine.dtype)
        loss = ((cosine - identity) ** 2).sum()
        return loss if self.paper_equations else loss / max(cosine.shape[0], 1)

    @torch.no_grad()
    def prune(self, utilization: torch.Tensor, sensitivity: float) -> list[int]:
        if utilization.shape != (self.max_prototypes,):
            raise ValueError("utilization has the wrong shape")
        active_indices = torch.nonzero(self.active_prototypes, as_tuple=False).squeeze(-1)
        if active_indices.numel() <= self.min_prototypes:
            return []
        candidates = active_indices[utilization[active_indices] <= sensitivity]
        max_prunable = int(active_indices.numel() - self.min_prototypes)
        if candidates.numel() > max_prunable:
            order = torch.argsort(utilization[candidates])
            candidates = candidates[order[:max_prunable]]
        self.active_prototypes[candidates] = False
        return candidates.cpu().tolist()


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, orthogonal_init: bool = True) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        if orthogonal_init:
            self.apply(_init_module)
            nn.init.orthogonal_(self.network[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)

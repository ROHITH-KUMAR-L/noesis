"""PPO controller network and the deployable policy (plan.md §7 Phase 6, prd.md FR-6).

* item encoder: MLP (2 × 128, ReLU) over [item features ‖ global features];
* context: masked mean + max pooling over item embeddings, ‖ global → MLP;
* gate head: {NO_EDIT, EDIT} from the context; EDIT is masked (probability 0)
  when no item is a candidate;
* item head: per item, 3 logits (KEEP, COMPRESS, PRUNE) from [item ‖ context]
  with invalid actions set to −inf, so their probability is exactly 0;
* value head: from the context.

Joint log-probability: ``log π(gate) + 1[EDIT] · Σ_candidates log π(a_i)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from memory_rl.controller.features import (
    ACTIONS,
    GLOBAL_FEATURES,
    ITEM_FEATURES,
    MAX_ITEMS,
    SCHEMA_VERSION,
    Observation,
    build_observation,
)
from memory_rl.envs.policy_api import ControllerState, Decision

NEG_INF = -1e9  # used as -inf; exp underflows to exactly 0 in float32


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NetConfig:
    hidden: int = 128
    #: initial logit bias of KEEP over COMPRESS/PRUNE (p(KEEP) ≈ 0.92 at 3.0), so an
    #: exploratory EDIT changes a few items instead of a third of the history
    keep_bias: float = 3.0
    #: initial logit bias of NO_EDIT over EDIT (p(EDIT) ≈ 0.27 at 1.0)
    no_edit_bias: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ObsBatch:
    items: torch.Tensor  # [B, M, F]
    present: torch.Tensor  # [B, M] bool
    action_mask: torch.Tensor  # [B, M, 3] bool
    global_features: torch.Tensor  # [B, G]

    @property
    def candidate(self) -> torch.Tensor:
        return self.action_mask.any(dim=-1)

    @classmethod
    def from_observations(cls, observations: list[Observation]) -> ObsBatch:
        return cls(
            items=torch.as_tensor(np.stack([o.items for o in observations])),
            present=torch.as_tensor(np.stack([o.item_present for o in observations])),
            action_mask=torch.as_tensor(np.stack([o.action_mask for o in observations])),
            global_features=torch.as_tensor(np.stack([o.global_features for o in observations])),
        )

    def index(self, idx: torch.Tensor) -> ObsBatch:
        return ObsBatch(
            self.items[idx], self.present[idx], self.action_mask[idx], self.global_features[idx]
        )


@dataclass(slots=True)
class PolicyOutput:
    gate_logits: torch.Tensor  # [B, 2] (masked)
    item_logits: torch.Tensor  # [B, M, 3] (masked)
    value: torch.Tensor  # [B]


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


class ControllerNet(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        h = self.config.hidden
        self.item_encoder = nn.Sequential(
            nn.Linear(ITEM_FEATURES + GLOBAL_FEATURES, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()
        )
        self.context = nn.Sequential(nn.Linear(2 * h + GLOBAL_FEATURES, h), nn.ReLU())
        self.gate_head = _mlp(h, h, 2)
        self.item_head = _mlp(2 * h, h, len(ACTIONS))
        self.value_head = _mlp(h, h, 1)
        with torch.no_grad():  # exploration-friendly initialisation (documented above)
            gate_out = self.gate_head[-1]
            item_out = self.item_head[-1]
            assert isinstance(gate_out, nn.Linear) and isinstance(item_out, nn.Linear)
            gate_out.weight.mul_(0.1)
            gate_out.bias.copy_(torch.tensor([self.config.no_edit_bias, 0.0]))
            item_out.weight.mul_(0.1)
            item_out.bias.copy_(torch.tensor([self.config.keep_bias, 0.0, 0.0]))

    def forward(self, obs: ObsBatch) -> PolicyOutput:
        b, m, _ = obs.items.shape
        g = obs.global_features
        x = torch.cat([obs.items, g.unsqueeze(1).expand(b, m, g.shape[-1])], dim=-1)
        emb = self.item_encoder(x)  # [B, M, H]
        present = obs.present.unsqueeze(-1).float()
        count = present.sum(dim=1).clamp(min=1.0)
        mean = (emb * present).sum(dim=1) / count
        masked = emb.masked_fill(~obs.present.unsqueeze(-1), NEG_INF)
        maxpool = masked.max(dim=1).values
        any_item = obs.present.any(dim=1, keepdim=True)
        maxpool = torch.where(any_item, maxpool, torch.zeros_like(maxpool))
        ctx = self.context(torch.cat([mean, maxpool, g], dim=-1))  # [B, H]

        gate_logits = self.gate_head(ctx)
        can_edit = obs.candidate.any(dim=1)
        gate_mask = torch.stack([torch.ones_like(can_edit), can_edit], dim=-1)
        gate_logits = gate_logits.masked_fill(~gate_mask, NEG_INF)

        item_in = torch.cat([emb, ctx.unsqueeze(1).expand(b, m, ctx.shape[-1])], dim=-1)
        item_logits = self.item_head(item_in)
        # Non-candidates get a dummy all-KEEP distribution (never used in log-probs).
        safe_mask = obs.action_mask.clone()
        safe_mask[..., 0] |= ~obs.candidate
        item_logits = item_logits.masked_fill(~safe_mask, NEG_INF)
        value = self.value_head(ctx).squeeze(-1)
        return PolicyOutput(gate_logits, item_logits, value)


@dataclass(slots=True)
class JointEval:
    log_prob: torch.Tensor  # [B]
    entropy: torch.Tensor  # [B]
    value: torch.Tensor  # [B]


def evaluate_actions(
    out: PolicyOutput, obs: ObsBatch, gate: torch.Tensor, items: torch.Tensor
) -> JointEval:
    gate_logp_all = torch.log_softmax(out.gate_logits, dim=-1)
    gate_lp = gate_logp_all.gather(1, gate.unsqueeze(1)).squeeze(1)
    gate_p = gate_logp_all.exp()
    gate_ent = -(gate_p * gate_logp_all.clamp(min=-50)).sum(-1)

    item_logp_all = torch.log_softmax(out.item_logits, dim=-1)  # [B, M, 3]
    item_lp = item_logp_all.gather(2, items.unsqueeze(-1)).squeeze(-1)  # [B, M]
    cand = obs.candidate.float()
    item_p = item_logp_all.exp()
    item_ent = -(item_p * item_logp_all.clamp(min=-50)).sum(-1)  # [B, M]
    edit = (gate == 1).float()
    log_prob = gate_lp + edit * (item_lp * cand).sum(-1)
    entropy = gate_ent + edit * (item_ent * cand).sum(-1) / cand.sum(-1).clamp(min=1.0)
    return JointEval(log_prob, entropy, out.value)


@torch.no_grad()
def act(
    net: ControllerNet, obs: Observation, *, greedy: bool, generator: torch.Generator | None
) -> tuple[int, np.ndarray, float, float]:
    """Sample (or argmax) one joint action. Returns gate, item actions, log-prob, value."""
    batch = ObsBatch.from_observations([obs])
    out = net(batch)
    if greedy:
        gate = out.gate_logits.argmax(-1)
        items = out.item_logits.argmax(-1)
    else:
        gate = torch.multinomial(torch.softmax(out.gate_logits, -1), 1, generator=generator)[:, 0]
        probs = torch.softmax(out.item_logits[0], -1)  # [M, 3]
        items = torch.multinomial(probs, 1, generator=generator)[:, 0].unsqueeze(0)
    joint = evaluate_actions(out, batch, gate, items)
    return int(gate[0]), items[0].numpy().copy(), float(joint.log_prob[0]), float(out.value[0])


# --- checkpoints & deployable policy ----------------------------------------------------


def save_checkpoint(path: str | Path, net: ControllerNet, meta: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "max_items": MAX_ITEMS,
            "net_config": net.config.to_dict(),
            "state_dict": net.state_dict(),
            "meta": dict(meta),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> tuple[ControllerNet, dict[str, Any]]:
    try:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
    except FileNotFoundError as exc:
        raise CheckpointError(f"checkpoint not found: {path}") from exc
    if data.get("schema_version") != SCHEMA_VERSION or data.get("max_items") != MAX_ITEMS:
        raise CheckpointError(
            f"checkpoint {path} uses feature schema {data.get('schema_version')!r} "
            f"(max_items={data.get('max_items')}), but this code expects {SCHEMA_VERSION!r} "
            f"(max_items={MAX_ITEMS}); retrain or check out the matching commit"
        )
    net = ControllerNet(NetConfig(**data["net_config"]))
    net.load_state_dict(data["state_dict"])
    net.eval()
    return net, dict(data["meta"])


class PPOMemoryPolicy:
    """A trained controller as a ``MemoryPolicy`` (greedy by default, for evaluation)."""

    def __init__(self, net: ControllerNet, *, name: str = "ppo", greedy: bool = True) -> None:
        self.net = net
        self.name = name
        self.greedy = greedy
        self._gen = torch.Generator().manual_seed(0)

    def reset(self, seed: int) -> None:
        self._gen = torch.Generator().manual_seed(seed)

    def decide(self, state: ControllerState) -> Decision:
        from memory_rl.controller.env import decision_from_arrays

        obs = build_observation(state)
        gate, items, _, _ = act(self.net, obs, greedy=self.greedy, generator=self._gen)
        return decision_from_arrays(obs, gate, items)


def load_policy(path: str | Path, *, greedy: bool = True) -> PPOMemoryPolicy:
    net, meta = load_checkpoint(path)
    name = f"ppo_{meta.get('variant', 'cache')}"
    return PPOMemoryPolicy(net, name=name, greedy=greedy)

"""Phase 6 PPO tests (plan.md §7 Phase 6). Skipped only if the [rl] extra is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional extra [rl] (torch) not installed")

from memory_rl.controller.env import ControllerEnv  # noqa: E402
from memory_rl.controller.features import MAX_ITEMS, Observation  # noqa: E402
from memory_rl.controller.policy import (  # noqa: E402
    CheckpointError,
    ControllerNet,
    ObsBatch,
    PPOMemoryPolicy,
    act,
    evaluate_actions,
    load_checkpoint,
    load_policy,
    save_checkpoint,
)
from memory_rl.controller.ppo import (  # noqa: E402
    NonFiniteError,
    PPOConfig,
    collect,
    compute_gae,
    update,
)
from memory_rl.controller.reward import RewardConfig  # noqa: E402
from memory_rl.cost import CacheAwareCostMeter, get_profile  # noqa: E402
from memory_rl.envs.agent_loop import AgentEpisode, LoopConfig  # noqa: E402
from memory_rl.envs.toy import KeyHuntEnv  # noqa: E402
from memory_rl.llm import ScriptedKeyHuntLLM  # noqa: E402
from memory_rl.memory import TestTokenizer  # noqa: E402

PROFILE = get_profile("anthropic-sonnet-5-5")


def factory(task_id: str, seed: int) -> AgentEpisode:
    return AgentEpisode(
        KeyHuntEnv(), task_id, seed=seed, llm=ScriptedKeyHuntLLM(),
        meter=CacheAwareCostMeter(PROFILE), tokenizer=TestTokenizer(), profile=PROFILE,
        config=LoopConfig(context_limit_tokens=20_000),
    )  # fmt: skip


def mid_episode_obs(turns: int = 4) -> Observation:
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    obs = cenv.reset("keyhunt-train-0002", 0)
    for _ in range(turns):
        nxt, *_ = cenv.step(0, np.zeros(MAX_ITEMS, dtype=int))
        assert nxt is not None
        obs = nxt
    return obs


def test_shapes_and_finiteness() -> None:
    obs = mid_episode_obs()
    batch = ObsBatch.from_observations([obs, obs])
    out = ControllerNet()(batch)
    assert out.gate_logits.shape == (2, 2)
    assert out.item_logits.shape == (2, MAX_ITEMS, 3)
    assert out.value.shape == (2,)
    for tensor in (out.gate_logits, out.item_logits, out.value):
        assert torch.isfinite(tensor).all()


def test_masked_actions_have_zero_probability_and_are_never_sampled() -> None:
    obs = mid_episode_obs()
    net = ControllerNet()
    last = net.item_head[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():  # make masked actions maximally attractive before masking
        last.bias.copy_(torch.tensor([-20.0, 20.0, 20.0]))
        probs = torch.softmax(net(ObsBatch.from_observations([obs])).item_logits[0], -1)
    invalid = ~torch.as_tensor(obs.action_mask) & torch.as_tensor(obs.candidate)[:, None]
    assert float(probs[invalid].sum()) == 0.0
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        gate, items, _, _ = act(net, obs, greedy=False, generator=gen)
        for slot in np.flatnonzero(obs.candidate):
            assert obs.action_mask[slot, items[slot]]


def test_gate_edit_masked_without_candidates() -> None:
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    first = cenv.reset("keyhunt-train-0002", 0)
    assert not first.candidate.any()
    logits = ControllerNet()(ObsBatch.from_observations([first])).gate_logits
    assert float(torch.softmax(logits, -1)[0, 1]) == 0.0


def test_joint_log_prob_is_gate_plus_items() -> None:
    obs = mid_episode_obs()
    net = ControllerNet()
    batch = ObsBatch.from_observations([obs])
    out = net(batch)
    items = torch.zeros(1, MAX_ITEMS, dtype=torch.long)
    cand = np.flatnonzero(obs.candidate)
    items[0, cand[0]] = 2  # PRUNE one candidate
    edit = evaluate_actions(out, batch, torch.tensor([1]), items)
    no_edit = evaluate_actions(out, batch, torch.tensor([0]), items)
    gate_lp = torch.log_softmax(out.gate_logits, -1)[0]
    item_lp = torch.log_softmax(out.item_logits, -1)[0]
    expected = gate_lp[1] + sum(item_lp[s, items[0, s]] for s in cand)
    assert torch.isclose(edit.log_prob[0], expected, atol=1e-5)
    assert torch.isclose(no_edit.log_prob[0], gate_lp[0], atol=1e-6)


def test_gae_terminal_does_not_bootstrap() -> None:
    rewards = np.array([0.0, 1.0, 0.0, 2.0])
    values = np.array([0.5, 0.5, 9.0, 9.0])
    dones = np.array([False, True, False, True])
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
    assert ret[1] == pytest.approx(1.0)  # terminal: return = reward
    assert ret[0] == pytest.approx(1.0)  # no leakage from the next episode's values
    assert ret[3] == pytest.approx(2.0)
    with pytest.raises(NonFiniteError):
        compute_gae(np.array([np.nan]), np.array([0.0]), np.array([True]), 1.0, 0.95)


def test_checkpoint_roundtrip_and_schema_mismatch(tmp_path: Path) -> None:
    net = ControllerNet()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, net, {"variant": "cache", "seed": 3})
    loaded, meta = load_checkpoint(path)
    assert meta["seed"] == 3
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    policy = load_policy(path)
    assert isinstance(policy, PPOMemoryPolicy) and policy.name == "ppo_cache"

    data = torch.load(path, weights_only=False)
    data["schema_version"] = "old-schema"
    torch.save(data, path)
    with pytest.raises(CheckpointError, match="retrain"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")


def _short_run(seed: int) -> list[float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    net = ControllerNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    config = PPOConfig(episodes_per_update=2, epochs=1, minibatch_size=16)
    ids = list(KeyHuntEnv().task_ids("train"))[:5]
    out = []
    for _ in range(2):
        rollout = collect(cenv, net, ids, rng=rng, generator=gen, episodes=2)
        stats = update(net, opt, rollout, config, rng=rng)
        out += [stats["policy_loss"], stats["value_loss"], *rollout.episode_returns]
    return out


def test_seeded_short_run_is_reproducible_on_cpu() -> None:
    torch.set_num_threads(1)
    assert _short_run(7) == _short_run(7)


def test_trained_policy_runs_as_memory_policy(tmp_path: Path) -> None:
    path = tmp_path / "p.pt"
    save_checkpoint(path, ControllerNet(), {"variant": "token"})
    episode = factory("keyhunt-valid-0000", 0).run(load_policy(path))
    record = episode.record(method="ppo_token")
    assert record["terminal_reason"] in {"success", "step_limit", "context_overflow", "env_done"}

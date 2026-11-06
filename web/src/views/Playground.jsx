import { useEffect, useState } from "react";
import Ledger, { LedgerKey } from "../components/Ledger.jsx";
import { api, int, pct, usd } from "../lib.js";

const STRATEGIES = {
  window: "Keep only the last K items",
  trim_every_turn: "Trim the oldest item every call",
  compact_once: "One big compaction at call C",
};

export default function Playground({ meta }) {
  const [form, setForm] = useState({
    profile: "anthropic-sonnet-5-5",
    prefix: 1500,
    item: 400,
    calls: 12,
    strategy: "window",
    window: 4,
    compact_at: 8,
  });
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const id = setTimeout(() => {
      api("/api/playground", form)
        .then((d) => {
          setData(d);
          setError(null);
        })
        .catch((e) => setError(e.message));
    }, 150);
    return () => clearTimeout(id);
  }, [form]);

  const set = (key) => (e) =>
    setForm((f) => ({ ...f, [key]: e.target.type === "range" || e.target.type === "number" ? Number(e.target.value) : e.target.value }));

  const base = data?.never_prune;
  const alt = data?.[form.strategy];
  const scale = data ? Math.max(...[...base.calls, ...alt.calls].map((c) => c.prompt_tokens)) * 1.05 : 1;
  const tokenDelta = base && alt ? alt.total_tokens / base.total_tokens - 1 : 0;
  const costDelta = base && alt ? alt.total_cost / base.total_cost - 1 : 0;
  const profile = meta.profiles.find((p) => p.name === form.profile);

  return (
    <section aria-labelledby="play-title">
      <h2 id="play-title" className="lede">
        Shape a conversation and a pruning rule, then compare the bill with never pruning at all.
      </h2>
      <p className="sub">
        A protected prompt is followed by one new history item per call. Every number below comes from the same
        prefix-cache simulator the agent experiments use, with the provider prices recorded in the pricing profile.
      </p>
      <div className="play">
        <form className="panel" onSubmit={(e) => e.preventDefault()}>
          <label className="field" htmlFor="pg-profile">
            Pricing
            <select id="pg-profile" value={form.profile} onChange={set("profile")}>
              {meta.profiles.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.illustrative ? "Illustrative (round numbers)" : p.model}
                </option>
              ))}
            </select>
          </label>
          {profile ? (
            <p className="sub" style={{ fontSize: "0.82rem", marginTop: -6 }}>
              Input ${profile.input_per_mtok}/M tokens. Cache reads cost {profile.cache_read_multiplier}× that and cache
              writes {profile.cache_write_multiplier}×. Caching needs at least {int(profile.min_cacheable_tokens)} tokens
              {profile.block_size > 1 ? `, in blocks of ${profile.block_size}` : ""}. Prices from{" "}
              <a href={profile.source}>the provider</a>, read {profile.retrieved}.
            </p>
          ) : null}
          <Slider id="pg-prefix" label="Protected prompt" unit="tokens" min={0} max={6000} step={100} value={form.prefix} onChange={set("prefix")} />
          <Slider id="pg-item" label="Each history item" unit="tokens" min={50} max={2000} step={50} value={form.item} onChange={set("item")} />
          <Slider id="pg-calls" label="LLM calls" unit="" min={2} max={30} step={1} value={form.calls} onChange={set("calls")} />
          <label className="field" htmlFor="pg-strategy">
            Pruning rule
            <select id="pg-strategy" value={form.strategy} onChange={set("strategy")}>
              {Object.entries(STRATEGIES).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <Slider id="pg-window" label={form.strategy === "compact_once" ? "Items kept after the compaction" : "Items kept (K)"} unit="" min={1} max={15} step={1} value={form.window} onChange={set("window")} />
          {form.strategy === "compact_once" ? (
            <Slider id="pg-at" label="Compaction at call (C)" unit="" min={2} max={form.calls} step={1} value={Math.min(form.compact_at, form.calls)} onChange={set("compact_at")} />
          ) : null}
        </form>

        <div className="stack">
          {error ? <p className="error">{error}</p> : null}
          {base && alt ? (
            <>
              <div>
                <p className="claim">
                  {STRATEGIES[form.strategy]} sends{" "}
                  <span className={tokenDelta <= 0 ? "down" : "up"}>
                    {pct(Math.abs(tokenDelta))} {tokenDelta <= 0 ? "fewer" : "more"} tokens
                  </span>{" "}
                  and costs{" "}
                  <span className={costDelta <= 0 ? "down" : "up"}>
                    {pct(Math.abs(costDelta))} {costDelta <= 0 ? "less" : "more"}
                  </span>{" "}
                  than keeping everything.
                </p>
                <p className="sub">
                  Never pruning: {int(base.total_tokens)} tokens, {usd(base.total_cost)}. With the rule:{" "}
                  {int(alt.total_tokens)} tokens, {usd(alt.total_cost)}. Each call also produces 60 output tokens.
                </p>
              </div>
              <LedgerKey />
              <div className="duel">
                <div className="lane">
                  <h3 className="lane-title">Never prune</h3>
                  <Ledger calls={base.calls.map((c) => ({ prompt_tokens: c.prompt_tokens, cost: c }))} scale={scale} />
                </div>
                <div className="lane">
                  <h3 className="lane-title">{STRATEGIES[form.strategy]}</h3>
                  <Ledger calls={alt.calls.map((c) => ({ prompt_tokens: c.prompt_tokens, cost: c }))} scale={scale} />
                </div>
              </div>
            </>
          ) : (
            <p className="empty">Computing…</p>
          )}
        </div>
      </div>
    </section>
  );
}

function Slider({ id, label, unit, value, onChange, ...rest }) {
  return (
    <label className="field" htmlFor={id}>
      <span>
        {label}: <strong>{int(value)}</strong> {unit}
      </span>
      <input id={id} type="range" value={value} onChange={onChange} {...rest} />
    </label>
  );
}

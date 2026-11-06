import { useEffect, useMemo, useRef, useState } from "react";
import Ledger, { LedgerKey } from "../components/Ledger.jsx";
import PolicyPicker from "../components/PolicyPicker.jsx";
import { api, describePolicy, firstLine, int, pct, usd } from "../lib.js";

const REASONS = {
  success: "Solved",
  context_overflow: "Failed: context budget exceeded",
  step_limit: "Failed: ran out of steps",
  env_done: "Failed: wrong keys submitted",
};

function editMark(actions) {
  const values = Object.values(actions || {});
  const pruned = values.filter((a) => a === "prune").length;
  const compressed = values.filter((a) => a === "compress").length;
  const parts = [];
  if (pruned) parts.push(`pruned ${pruned}`);
  if (compressed) parts.push(`compressed ${compressed}`);
  return parts.join(", ");
}

function defaultRight(meta) {
  const ppo =
    meta.checkpoints.find((c) => c.variant === "cache" && c.lam === 1 && c.seed === 0) ??
    meta.checkpoints.find((c) => c.variant === "cache");
  if (ppo) return { method: "ppo_cache", knob: null, checkpoint: ppo.id };
  return { method: "threshold_compaction", knob: 3500, checkpoint: null };
}

export default function Replay({ meta }) {
  const longest = meta.tasks.reduce((a, b) => (b.keys > a.keys ? b : a), meta.tasks[0]);
  const [taskId, setTaskId] = useState(longest.id);
  const [profile, setProfile] = useState("anthropic-sonnet-5-5");
  const [left, setLeft] = useState({ method: "full_history", knob: null, checkpoint: null });
  const [right, setRight] = useState(() => defaultRight(meta));
  const [runs, setRuns] = useState({ left: null, right: null });
  const [error, setError] = useState(null);
  const [turn, setTurn] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    let live = true;
    setError(null);
    const body = (p) => ({ task_id: taskId, profile, ...p });
    Promise.all([api("/api/episode", body(left)), api("/api/episode", body(right))])
      .then(([l, r]) => {
        if (!live) return;
        setRuns({ left: l, right: r });
        setTurn(Math.max(l.turns.length, r.turns.length) - 1);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [taskId, profile, left, right]);

  const maxTurns = Math.max(runs.left?.turns.length ?? 0, runs.right?.turns.length ?? 0);
  const scale = useMemo(() => {
    const all = [runs.left, runs.right].filter(Boolean).flatMap((r) => r.turns.map((t) => t.prompt_tokens));
    return Math.max(...all, meta.context_limit) * 1.08;
  }, [runs, meta.context_limit]);

  useEffect(() => {
    if (!playing) return undefined;
    timer.current = setInterval(() => {
      setTurn((t) => {
        if (t + 1 >= maxTurns) {
          setPlaying(false);
          return t;
        }
        return t + 1;
      });
    }, 900);
    return () => clearInterval(timer.current);
  }, [playing, maxTurns]);

  const play = () => {
    if (turn + 1 >= maxTurns) setTurn(0);
    setPlaying((p) => !p);
  };

  return (
    <section aria-labelledby="replay-title">
      <h2 id="replay-title" className="lede">
        Same agent, same task, two memory policies. Each bar is one LLM call, split by how the provider bills it.
      </h2>
      <p className="sub">
        A memory policy runs before every call and may keep, compress or prune earlier steps. An edit changes the prompt
        from that point on, so the provider can no longer serve that part from cache and bills it again. Fewer tokens
        can still mean a bigger bill.
      </p>

      <div className="controls">
        <label className="field" htmlFor="task">
          Task (validation split)
          <select id="task" value={taskId} onChange={(e) => setTaskId(e.target.value)}>
            {meta.tasks.map((t) => (
              <option key={t.id} value={t.id}>
                {t.id.replace("keyhunt-valid-", "#")}, {t.keys} keys
              </option>
            ))}
          </select>
        </label>
        <label className="field" htmlFor="profile">
          Pricing
          <select id="profile" value={profile} onChange={(e) => setProfile(e.target.value)}>
            {meta.profiles
              .filter((p) => !p.illustrative)
              .map((p) => (
                <option key={p.name} value={p.name}>
                  {p.model} ({p.mode === "breakpoint" ? "explicit breakpoints" : "automatic caching"})
                </option>
              ))}
          </select>
        </label>
      </div>

      {error ? <p className="error">{error}</p> : null}
      {runs.left?.task ? (
        <p className="sub" style={{ marginBottom: 12 }}>
          <strong>{runs.left.task.instruction.replace("Task: ", "")}</strong> The protected prompt (instructions plus a
          worked example) is {int(runs.left.prefix_tokens)} tokens and is never edited.
        </p>
      ) : null}

      <div className="scrub">
        <button type="button" className="btn" onClick={play} disabled={!maxTurns}>
          {playing ? "Pause" : turn + 1 >= maxTurns ? "Replay from start" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(maxTurns - 1, 0)}
          value={turn}
          onChange={(e) => {
            setPlaying(false);
            setTurn(Number(e.target.value));
          }}
          aria-label="Call shown in the context panels"
        />
        <span className="scrub-label">
          call {turn + 1} of {maxTurns}
        </span>
      </div>
      <LedgerKey />

      <div className="duel">
        {[
          ["left", left, setLeft],
          ["right", right, setRight],
        ].map(([side, value, set]) => (
          <Lane
            key={side}
            side={side}
            meta={meta}
            policy={value}
            onPolicy={set}
            run={runs[side]}
            other={runs[side === "left" ? "right" : "left"]}
            scale={scale}
            turn={turn}
            onTurn={(t) => {
              setPlaying(false);
              setTurn(t);
            }}
          />
        ))}
      </div>
    </section>
  );
}

function Lane({ side, meta, policy, onPolicy, run, other, scale, turn, onTurn }) {
  const calls = (run?.turns ?? []).map((t) => ({ ...t, mark: editMark(t.actions) }));
  const shown = run ? Math.min(turn, run.turns.length - 1) : 0;
  const t = run?.turns[shown];
  const rec = run?.record;
  const cachedShare = rec ? rec.cached_tokens_total / rec.prompt_tokens_total : null;
  const diff = rec && other ? rec.total_cost / other.record.total_cost - 1 : null;

  return (
    <article className="lane" aria-label={`${side} policy`}>
      <div className="lane-head">
        <h3 className="lane-title">{describePolicy(policy)}</h3>
        {rec ? (
          <span className={`verdict ${rec.success ? "ok" : "fail"}`}>{REASONS[rec.terminal_reason] ?? rec.terminal_reason}</span>
        ) : null}
      </div>
      <div className="controls" style={{ margin: "12px 0 0" }}>
        <PolicyPicker meta={meta} value={policy} onChange={onPolicy} idPrefix={side} />
      </div>
      {rec ? (
        <div className="totals">
          <div>
            <div className="total-value">{usd(rec.total_cost)}</div>
            <div className="total-label">
              episode cost{diff != null && Math.abs(diff) > 0.005 ? `, ${diff > 0 ? "+" : ""}${pct(diff)} vs other` : ""}
            </div>
          </div>
          <div>
            <div className="total-value">{int(rec.prompt_tokens_total)}</div>
            <div className="total-label">prompt tokens sent</div>
          </div>
          <div>
            <div className="total-value">{pct(cachedShare)}</div>
            <div className="total-label">served from cache</div>
          </div>
          <div>
            <div className="total-value">{rec.edits}</div>
            <div className="total-label">edits ({int(rec.tokens_removed)} tokens removed)</div>
          </div>
        </div>
      ) : (
        <p className="empty">Running the episode…</p>
      )}
      {run ? <Ledger calls={calls} scale={scale} limit={run.context_limit} current={shown} onSelect={onTurn} /> : null}
      {t ? <Context turn={t} /> : null}
    </article>
  );
}

function Context({ turn }) {
  const changed = new Set(Object.keys(turn.actions || {}));
  return (
    <div className="context">
      <h4>
        What the model saw at call {turn.turn + 1}
        {turn.edit && changed.size ? `, after the policy ${editMark(turn.actions)}` : ""}
      </h4>
      <ul className="items">
        {turn.items.map((item) => (
          <li
            key={item.id}
            className={`item ${item.type} ${item.status}${changed.has(item.id) ? " changed" : ""}`}
            title={item.text}
          >
            <span className="item-kind">{item.type === "assistant" ? "agent" : "tool"}</span>
            <span className="item-text">
              {item.status === "pruned" ? "pruned: " : item.status === "compressed" ? "compressed: " : ""}
              {firstLine(item.text)}
            </span>
            <span className="item-tokens">{item.status === "pruned" ? "0" : int(item.tokens)}</span>
          </li>
        ))}
        {turn.items.length === 0 ? <li className="item">Only the protected prompt so far.</li> : null}
      </ul>
      <div className="said">
        {turn.llm_text ? (
          <div>
            <div className="said-label">The agent replied</div>
            <pre>{turn.llm_text}</pre>
          </div>
        ) : null}
        {turn.observation ? (
          <div>
            <div className="said-label">The tool returned</div>
            <pre>{turn.observation}</pre>
          </div>
        ) : null}
      </div>
    </div>
  );
}

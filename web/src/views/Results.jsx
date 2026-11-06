import { useEffect, useMemo, useState } from "react";
import { METHOD_COLORS, METHOD_LABELS, api, describePolicy, pct, usd } from "../lib.js";

const W = 760;
const H = 420;
const M = { top: 20, right: 30, bottom: 48, left: 58 };

function pareto(rows) {
  // Non-dominated: no other row has lower-or-equal cost and higher-or-equal success (one strictly).
  return rows.filter(
    (r) =>
      !rows.some(
        (o) =>
          o !== r &&
          o.cost_mean <= r.cost_mean &&
          o.success_mean >= r.success_mean &&
          (o.cost_mean < r.cost_mean || o.success_mean > r.success_mean),
      ),
  );
}

export default function Results() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [runName, setRunName] = useState(null);
  const [profile, setProfile] = useState(null);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    api("/api/results")
      .then((d) => {
        setData(d);
        const preferred = d.runs.find((r) => r.name.startsWith("final")) ?? d.runs[d.runs.length - 1];
        setRunName(preferred?.name ?? null);
      })
      .catch((e) => setError(e.message));
  }, []);

  const run = data?.runs.find((r) => r.name === runName);
  const profiles = useMemo(() => [...new Set((run?.rows ?? []).map((r) => r.pricing_profile))], [run]);
  useEffect(() => {
    if (profiles.length && !profiles.includes(profile)) setProfile(profiles[0]);
  }, [profiles, profile]);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="empty">Loading results…</p>;
  if (!data.runs.length)
    return (
      <p className="empty">
        No evaluation runs yet. Run <code>python -m memory_rl.evaluation.run_eval --config configs/evaluation.yaml</code>{" "}
        to create one.
      </p>
    );

  const rows = (run?.rows ?? [])
    .filter((r) => r.pricing_profile === profile)
    .map((r) => ({ ...r, family: r.method.startsWith("ppo") ? r.method.replace(/_lam.*/, "") : r.method }));
  const front = new Set(pareto(rows));
  const maxCost = Math.max(...rows.map((r) => r.cost_hi), 0.001) * 1.08;
  const minCost = Math.min(...rows.map((r) => r.cost_lo), maxCost) * 0.92;
  const x = (v) => M.left + ((v - minCost) / (maxCost - minCost)) * (W - M.left - M.right);
  const y = (v) => H - M.bottom - v * (H - M.top - M.bottom);
  const xticks = Array.from({ length: 5 }, (_, i) => minCost + ((maxCost - minCost) * i) / 4);
  const frontPts = [...front].sort((a, b) => a.cost_mean - b.cost_mean);
  const families = [...new Set(rows.map((r) => r.family))];

  return (
    <section aria-labelledby="results-title">
      <h2 id="results-title" className="lede">
        Task success against what the episodes actually cost, after caching discounts.
      </h2>
      <p className="sub">
        Each point is one method and setting, averaged over {run?.tasks} tasks and seeds {run?.seeds?.join(", ")} on the{" "}
        {run?.split} split. Whiskers are 95% bootstrap intervals over tasks. The dashed line joins the settings that no
        other setting beats on both success and cost.
      </p>
      <div className="controls">
        <label className="field" htmlFor="run">
          Evaluation run
          <select id="run" value={runName ?? ""} onChange={(e) => setRunName(e.target.value)}>
            {data.runs.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} ({r.split})
              </option>
            ))}
          </select>
        </label>
        <label className="field" htmlFor="res-profile">
          Pricing
          <select id="res-profile" value={profile ?? ""} onChange={(e) => setProfile(e.target.value)}>
            {profiles.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="chart-wrap">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Success rate against episode cost">
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line className="grid-line" x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} />
              <text className="tick" x={M.left - 8} y={y(v) + 4} textAnchor="end">
                {pct(v)}
              </text>
            </g>
          ))}
          {xticks.map((v) => (
            <text key={v} className="tick" x={x(v)} y={H - M.bottom + 18} textAnchor="middle">
              {usd(v)}
            </text>
          ))}
          <text className="axis-label" x={(W + M.left) / 2} y={H - 8} textAnchor="middle">
            Mean cost per episode, cache-adjusted
          </text>
          <text className="axis-label" transform={`translate(14 ${(H - M.bottom + M.top) / 2}) rotate(-90)`} textAnchor="middle">
            Tasks solved
          </text>
          {frontPts.length > 1 ? (
            <polyline className="frontier" points={frontPts.map((r) => `${x(r.cost_mean)},${y(r.success_mean)}`).join(" ")} />
          ) : null}
          {rows.map((r, i) => {
            const color = METHOD_COLORS[r.family] ?? "var(--ink-3)";
            const cx = x(r.cost_mean);
            const cy = y(r.success_mean);
            return (
              <g key={i} onMouseEnter={() => setHover({ r, cx, cy })} onMouseLeave={() => setHover(null)}>
                <line x1={x(r.cost_lo)} x2={x(r.cost_hi)} y1={cy} y2={cy} stroke={color} strokeWidth={1.5} opacity={0.6} />
                <line x1={cx} x2={cx} y1={y(r.success_lo)} y2={y(r.success_hi)} stroke={color} strokeWidth={1.5} opacity={0.6} />
                <circle cx={cx} cy={cy} r={front.has(r) ? 6 : 4.5} fill={color} stroke="var(--surface)" strokeWidth={2} />
                <circle cx={cx} cy={cy} r={14} fill="transparent" />
                {front.has(r) ? (
                  <text className="tick" x={cx + 9} y={cy - 9} style={{ fill: "var(--ink)" }}>
                    {shortLabel(r)}
                  </text>
                ) : null}
              </g>
            );
          })}
          {hover ? (
            <g pointerEvents="none" transform={`translate(${Math.min(hover.cx + 12, W - 230)} ${Math.max(hover.cy - 66, 4)})`}>
              <rect width={220} height={60} rx={6} fill="var(--ink)" />
              <text x={10} y={18} fill="var(--surface)" fontSize={12} fontWeight={700}>
                {shortLabel(hover.r)}
              </text>
              <text x={10} y={35} fill="var(--surface)" fontSize={11}>
                solved {pct(hover.r.success_mean)}, {usd(hover.r.cost_mean)} per episode
              </text>
              <text x={10} y={51} fill="var(--surface)" fontSize={11}>
                {hover.r.edits_mean.toFixed(1)} edits, {pct(hover.r.cached_fraction)} cached
              </text>
            </g>
          ) : null}
        </svg>
        <div className="legend">
          {families.map((f) => (
            <span key={f}>
              <i className="dot" style={{ background: METHOD_COLORS[f] ?? "var(--ink-3)" }} />
              {METHOD_LABELS[f] ?? f}
            </span>
          ))}
        </div>
      </div>

      <table className="rows">
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Setting</th>
            <th scope="col">Solved (95% CI)</th>
            <th scope="col">Cost per episode (95% CI)</th>
            <th scope="col">Prompt tokens</th>
            <th scope="col">From cache</th>
            <th scope="col">Edits</th>
            <th scope="col">Cost per solved task</th>
          </tr>
        </thead>
        <tbody>
          {[...rows]
            .sort((a, b) => a.cost_mean - b.cost_mean)
            .map((r, i) => (
              <tr key={i} className={front.has(r) ? "frontier-row" : ""}>
                <td>{METHOD_LABELS[r.family] ?? r.method}</td>
                <td>{r.knob === "-" ? "" : r.knob}</td>
                <td>
                  {pct(r.success_mean)} ({pct(r.success_lo)}–{pct(r.success_hi)})
                </td>
                <td>
                  {usd(r.cost_mean)} ({usd(r.cost_lo)}–{usd(r.cost_hi)})
                </td>
                <td>{Math.round(r.tokens_mean).toLocaleString("en-US")}</td>
                <td>{pct(r.cached_fraction)}</td>
                <td>{r.edits_mean.toFixed(1)}</td>
                <td>{Number.isFinite(r.cost_per_success) ? usd(r.cost_per_success) : "no successes"}</td>
              </tr>
            ))}
        </tbody>
      </table>
      <p className="honesty">
        {run?.note}. LLM: {run?.llm}. Code version {run?.git_commit}.
      </p>
    </section>
  );
}

function shortLabel(r) {
  return describePolicy({ method: r.family, knob: r.knob === "-" || r.family.startsWith("ppo") ? null : r.knob }) +
    (r.family.startsWith("ppo") ? ` ${r.knob.replace(/^ppo_(cache|token)_/, "").replace(/_anthropic.*|_openai.*/, "")}` : "");
}

import { useState } from "react";
import { int, usd } from "../lib.js";

/**
 * One row per LLM call. Bar length = prompt tokens (shared scale across lanes);
 * segments = what the provider served from cache, wrote to cache, and billed in full.
 */
export default function Ledger({ calls, scale, limit, current, onSelect, label = "Call" }) {
  const [hover, setHover] = useState(null);
  const width = (n) => `${Math.max((n / scale) * 100, 0)}%`;

  return (
    <div className="ledger" onMouseLeave={() => setHover(null)}>
      <div className="ledger-body">
        {limit ? (
          <div className="limit-line" style={{ left: `calc(3.2ch + 8px + (100% - 3.2ch - 8px) * ${limit / scale})` }}>
            <span className="limit-label">context budget {int(limit)}</span>
          </div>
        ) : null}
        {calls.map((call, i) => {
          const cost = call.cost;
          const state = current == null ? "" : i === current ? "current" : i > current ? "future" : "";
          return (
            <button
              type="button"
              key={i}
              className={`ledger-row ${state}`}
              onClick={() => onSelect?.(i)}
              onMouseMove={(e) => setHover({ i, x: e.clientX, y: e.clientY })}
              onFocus={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setHover({ i, x: r.left + 40, y: r.bottom });
              }}
              onBlur={() => setHover(null)}
              aria-label={`${label} ${i + 1}: ${int(call.prompt_tokens)} prompt tokens${cost ? `, ${int(cost.cached_tokens)} cached, ${usd(cost.total_cost)}` : ", not sent (context overflow)"}`}
            >
              <span className="ledger-n">{i + 1}</span>
              <span className="bar-track">
                {cost ? (
                  <span className="bar" style={{ width: width(call.prompt_tokens) }}>
                    {cost.cached_tokens > 0 && <span className="seg cached" style={{ flex: cost.cached_tokens }} />}
                    {cost.written_tokens > 0 && <span className="seg written" style={{ flex: cost.written_tokens }} />}
                    {cost.uncached_tokens > 0 && <span className="seg uncached" style={{ flex: cost.uncached_tokens }} />}
                  </span>
                ) : (
                  <span className="bar overflow" style={{ width: width(call.prompt_tokens) }} />
                )}
                {call.mark ? (
                  <span className="edit-mark" style={{ left: width(call.prompt_tokens) }}>
                    {call.mark}
                  </span>
                ) : null}
              </span>
            </button>
          );
        })}
      </div>
      {hover && calls[hover.i] ? <HoverCard call={calls[hover.i]} n={hover.i + 1} x={hover.x} y={hover.y} label={label} /> : null}
    </div>
  );
}

function HoverCard({ call, n, x, y, label }) {
  const c = call.cost;
  const left = Math.min(x + 14, window.innerWidth - 300);
  return (
    <div className="hovercard" style={{ left, top: y + 14 }} role="tooltip">
      <b>
        {label} {n}: {int(call.prompt_tokens)} prompt tokens
      </b>
      {c ? (
        <>
          <div>Served from cache: {int(c.cached_tokens)} ({usd(c.cache_read_cost)})</div>
          <div>Written to cache: {int(c.written_tokens)} ({usd(c.cache_write_cost)})</div>
          <div>Billed in full: {int(c.uncached_tokens)} ({usd(c.input_cost)})</div>
          <div>Output: {int(c.output_tokens)} tokens ({usd(c.output_cost)})</div>
          <div>
            <b>Call cost {usd(c.total_cost)}</b>
          </div>
        </>
      ) : (
        <div>Prompt exceeds the context budget, so it was never sent. The episode fails here.</div>
      )}
      {call.mark ? <div>Edit before this call: {call.mark}</div> : null}
    </div>
  );
}

export function LedgerKey() {
  return (
    <div className="key" aria-hidden="true">
      <span>
        <i className="swatch" style={{ background: "var(--cached)" }} /> served from cache
      </span>
      <span>
        <i className="swatch" style={{ background: "var(--written)" }} /> written to cache
      </span>
      <span>
        <i className="swatch" style={{ background: "var(--uncached)" }} /> billed in full
      </span>
      <span>
        <i className="swatch" style={{ background: "repeating-linear-gradient(135deg, var(--fail) 0 2px, transparent 2px 5px)" }} /> over budget, not sent
      </span>
    </div>
  );
}

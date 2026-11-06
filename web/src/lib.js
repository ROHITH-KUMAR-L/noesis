export async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${path} failed: ${detail}`);
  }
  return res.json();
}

export const usd = (v) => {
  if (v == null || Number.isNaN(v)) return "—";
  if (v === 0) return "$0";
  if (v < 0.001) return `$${(v * 1000).toFixed(3)}m`;
  return `$${v.toFixed(4)}`;
};
export const int = (v) => (v == null ? "—" : Math.round(v).toLocaleString("en-US"));
export const pct = (v, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);

/** Fixed categorical order (validated palette); colour follows the method, never its rank. */
export const METHOD_COLORS = {
  full_history: "var(--s1)",
  sliding_window: "var(--s2)",
  obs_masking: "var(--s3)",
  threshold_compaction: "var(--s4)",
  cadoc_rule: "var(--s5)",
  random: "var(--s6)",
  ppo_cache: "var(--s7)",
  ppo_token: "var(--s8)",
};

export const METHOD_LABELS = {
  full_history: "Full history",
  sliding_window: "Sliding window",
  obs_masking: "Observation masking",
  threshold_compaction: "Threshold compaction",
  cadoc_rule: "CADOC rule",
  random: "Random",
  ppo_cache: "PPO, cache-aware reward",
  ppo_token: "PPO, token reward",
};

export const KNOB_NAMES = {
  sliding_window: "keep last K turns",
  obs_masking: "mask after M turns",
  threshold_compaction: "compact above T tokens",
  cadoc_rule: "pending after M turns",
  random: "edit probability",
};

export function describePolicy(p) {
  const label = METHOD_LABELS[p.method] ?? p.method;
  if (p.method.startsWith("ppo")) return label;
  if (p.knob == null) return label;
  return `${label} (${KNOB_NAMES[p.method] ?? "knob"} ${p.knob})`;
}

export function firstLine(text, max = 120) {
  const line = (text || "").split("\n").find((l) => l.trim()) ?? "";
  return line.length > max ? `${line.slice(0, max)}…` : line;
}

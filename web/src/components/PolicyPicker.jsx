import { KNOB_NAMES, METHOD_LABELS } from "../lib.js";

/** Choose a method and its knob (baselines) or checkpoint (PPO). */
export default function PolicyPicker({ meta, value, onChange, idPrefix }) {
  const methods = [
    ...meta.methods.map((m) => m.id),
    ...(meta.checkpoints.some((c) => c.variant === "cache") ? ["ppo_cache"] : []),
    ...(meta.checkpoints.some((c) => c.variant === "token") ? ["ppo_token"] : []),
  ];
  const spec = meta.methods.find((m) => m.id === value.method);
  const variant = value.method.startsWith("ppo") ? value.method.split("_")[1] : null;
  const runs = variant ? meta.checkpoints.filter((c) => c.variant === variant) : [];

  const setMethod = (method) => {
    if (method.startsWith("ppo")) {
      const v = method.split("_")[1];
      const pick =
        meta.checkpoints.find((c) => c.variant === v && c.lam === 1 && c.seed === 0) ??
        meta.checkpoints.find((c) => c.variant === v);
      onChange({ method, knob: null, checkpoint: pick?.id ?? null });
    } else {
      const s = meta.methods.find((m) => m.id === method);
      onChange({ method, knob: s?.knobs?.[0] ?? null, checkpoint: null });
    }
  };

  return (
    <>
      <label className="field" htmlFor={`${idPrefix}-method`}>
        Policy
        <select id={`${idPrefix}-method`} value={value.method} onChange={(e) => setMethod(e.target.value)}>
          {methods.map((m) => (
            <option key={m} value={m}>
              {METHOD_LABELS[m] ?? m}
            </option>
          ))}
        </select>
      </label>
      {spec && spec.knobs.length > 0 ? (
        <label className="field" htmlFor={`${idPrefix}-knob`}>
          {KNOB_NAMES[value.method] ?? "Setting"}
          <select
            id={`${idPrefix}-knob`}
            value={value.knob ?? ""}
            onChange={(e) => onChange({ ...value, knob: Number(e.target.value) })}
          >
            {spec.knobs.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {variant ? (
        <label className="field" htmlFor={`${idPrefix}-ckpt`}>
          Trained run
          <select
            id={`${idPrefix}-ckpt`}
            value={value.checkpoint ?? ""}
            onChange={(e) => onChange({ ...value, checkpoint: e.target.value })}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                λ = {r.lam}, seed {r.seed}
              </option>
            ))}
          </select>
        </label>
      ) : null}
    </>
  );
}

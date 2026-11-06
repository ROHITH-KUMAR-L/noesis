# Baselines (Phase 5)

All baselines implement `MemoryPolicy` (`src/memory_rl/envs/policy_api.py`), run through the
same `AgentEpisode` harness, and only choose actions inside the masks. Grids live in
`configs/baselines.yaml`.

| Method | Rule | Knob (grid) |
|---|---|---|
| `full_history` | never edit | — |
| `sliding_window` | prune items older than the last *K* turns | K ∈ {1,2,3,4} |
| `obs_masking` (must-beat) | replace tool outputs older than *M* turns with `[old observation omitted]`; keep thoughts/actions | M ∈ {0,1,2,3} |
| `threshold_compaction` | no edits until the prompt exceeds *T* tokens; then compress every compressible item and prune the oldest tool outputs down to ≈ *T/2* | T ∈ {2500,3000,3500,4000} |
| `cadoc_rule` | CADOC online commit rule (below) on tool outputs older than *M* turns | M ∈ {1,2,3} |
| `random` | random gate (p) and random valid per-item actions | p ∈ {0.1,0.3} |
| `llmlingua` | LLMLingua-2 once on tool outputs older than *M* turns | optional extra, not installed by default |

Observation masking needs a different "compressor" (a fixed placeholder). It is passed
to the harness as that policy's `compressor`. Every other method uses the shared
`LineTruncationCompressor`.

## CADOC-style rule: re-implementation notes

Source: J. Yao, Z. Zhou, Z.-Q. J. Xu, *CADOC: Cache-Aware Dynamic Object Context for
Long-Horizon Agents*, arXiv:2609.37012v1 (29 Sep 2026), Section 3. The equations below
were read from the arXiv HTML on 2026-11-06. Appendices A, B and F were not read; check them
against the PDF before the report.

Costs are in uncached-token units, and cached tokens are weighted by `w_r` (Eq. 1:
`C = Σ_t N_t + w_r R_t`). We set `w_r` to the pricing profile's cache-read multiplier.

| Paper | Here |
|---|---|
| pending objects | tool outputs older than `M` turns that the fixed compressor can still shorten |
| `G_t = L_keep − L_commit` (Eq. 11) | Σ `compression_gain` of the pending items |
| ledger `ΔW_t = w_r G_t` (Eq. 12) | added after every request that still carries them raw; reset on commit |
| `Q_t = F_0 + (1 − w_r) H_t` (Eq. 13) | `F_0 = 0` (as in the paper's replay); `H_t` = tokens after the newest pending item and before the cache frontier (the end of the previous prompt) |
| commit iff `W_t + w_r G_t ≥ Q_t` (Eq. 14) | identical; commit = COMPRESS all pending items |

Deviations:

1. CADOC replaces objects with retrievable *Cards*. This environment has no retrieval, so
   the replacement is the shared fixed compressor. The originals stay only in the archive.
2. The paper notes that hard capacity limits can force an earlier commit. We force one when
   the prompt exceeds `guard_fill = 0.9` of the context limit.
3. The paper's stationary analysis (Eqs. 3–9: optimal batch `b* = √(2Q_h/r)`) is not used
   directly. The online rule (Eq. 14) is what runs.

`tests/test_baselines.py::test_cadoc_rule_follows_eq14_ledger` checks the ledger and the
commit condition turn by turn.

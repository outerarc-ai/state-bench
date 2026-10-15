# Snapshot policies on recorded agent traces (simulated)

**Result (simulated): on 120 recorded Claude Code sessions, the policy choice matters
only when the state pool is short. With an unlimited pool, SGLang's current placement
recomputes 3.8–5.1% of prompt tokens, within 0.3 points of the offline reference. At 64
slots it recomputes 20–40%, while value-based retention (`boundary_value`) recomputes
6.4–8.3%, and 6.0–8.3% with its demand point, the lowest of the online policies in all
six cases. When replies are not replayed token for token, Marconi-style admission recomputes
96% at every pool size.** These are simulator outputs on recorded traces, not
measurements of an engine. Configuration: [config.yaml](config.yaml).

## Setup

- Traces: three disjoint subsets of 40 kv-cache-tester traces (sessions 1–40, 41–80,
  81–120 in file order) at commit `94f6046`, imported with `state-bench import weka`.
  1,368 to 4,369 requests and 257M to 513M prompt tokens per subset; median prompt
  104k–131k tokens. Session `i` starts at `60 * i` seconds (assumption).
- Simulator: as in the [synthetic comparison](../2026-10-13-sim-policy-comparison/summary.md):
  sequential requests, unlimited KV, `chunked_prefill_size` 8192 with the hashed first
  chunk cut. The imported traces have no message boundaries, so `prompt_end_anchor` and
  `boundary_value` place at prompt ends (and reply ends) only.
- Both reply round-trip assumptions, because the traces cannot tell them apart.

## Results

Recomputed share of prompt tokens (%), subsets 1 / 2 / 3. Lowest online policy per row in
bold; `offline_oracle` is a reference that knows the future.

**Replies replayed in the next prompt:**

| Policy | Unlimited | 256 slots | 64 slots | 16 slots |
| --- | --- | --- | --- | --- |
| sglang_current | **3.92** / **4.86** / **3.76** | **4.95** / 8.61 / 7.29 | 21.65 / 39.98 / 36.62 | 62.03 / 78.90 / 70.35 |
| prompt_end_anchor | 4.97 / 5.28 / 4.78 | 5.60 / 6.24 / 5.90 | 7.34 / 9.89 / 9.06 | 17.87 / 23.30 / 23.16 |
| marconi_admission | 4.56 / 5.36 / 4.56 | 5.07 / 6.30 / 5.38 | 6.57 / 9.79 / 8.62 | 18.24 / 24.41 / 23.26 |
| pin_recent | **3.92** / **4.86** / **3.76** | **4.95** / 8.60 / 7.27 | 8.77 / 13.05 / 12.73 | 27.03 / 39.58 / 35.00 |
| boundary_value | 4.70 / 5.08 / 4.57 | 5.43 / 5.81 / 5.44 | 6.81 / 8.28 / 7.91 | 12.39 / **20.01** / **19.89** |
| boundary_value:demand=true | 4.29 / 4.96 / 4.34 | 5.02 / **5.77** / **5.11** | **6.39** / **8.26** / **7.55** | **12.28** / 20.11 / 20.04 |
| offline_oracle | 3.75 / 4.70 / 3.66 | 3.93 / 4.74 / 4.10 | 4.43 / 5.04 / 4.23 | 5.55 / 6.72 / 5.26 |

**Replies not replayed:**

| Policy | Unlimited | 256 slots | 64 slots | 16 slots |
| --- | --- | --- | --- | --- |
| sglang_current | **4.21** / **5.06** / **3.96** | 5.02 / 7.94 / 7.08 | 19.84 / 38.31 / 35.97 | 59.97 / 77.24 / 69.96 |
| prompt_end_anchor | 4.97 / 5.28 / 4.78 | 5.60 / 6.24 / 5.90 | 7.34 / 9.89 / 9.06 | 17.87 / 23.30 / 23.16 |
| marconi_admission | 95.99 / 96.67 / 95.54 | 95.99 / 96.67 / 95.54 | 95.99 / 96.67 / 95.54 | 95.99 / 96.67 / 95.54 |
| pin_recent | **4.21** / **5.06** / **3.96** | **5.00** / 7.94 / 7.06 | 8.70 / 12.32 / 12.67 | 25.61 / 36.21 / 33.08 |
| boundary_value | 4.97 / 5.28 / 4.78 | 5.50 / 5.54 / 5.29 | 6.44 / 7.29 / 6.93 | 10.15 / **13.82** / **14.06** |
| boundary_value:demand=true | 4.56 / 5.16 / 4.54 | 5.04 / **5.41** / **5.09** | **6.03** / **7.17** / **6.70** | **10.03** / 14.12 / 14.55 |
| offline_oracle | 3.96 / 4.92 / 3.82 | 3.96 / 4.92 / 3.82 | 3.96 / 4.92 / 3.82 | 4.26 / 5.93 / 4.82 |

Snapshot writes per request at 64 slots, subset 1, replies replayed: `sglang_current`
4.7, `pin_recent` 2.8, `boundary_value` 1.8, `marconi_admission` 0.93, `offline_oracle`
0.95. Full output, including writes, rejections, and evictions: `raw/`.

## Reading

- Most of what any policy can save is already saved with an unlimited pool: the offline
  reference still recomputes 3.7–4.9%, which includes first requests and the rewritten
  parts of prompts that no snapshot can cover. With
  enough slots, SGLang's current placement is within 0.3 points of it.
- The gap opens when the pool holds fewer snapshots than there are active chains. With
  40 overlapping sessions, SGLang's LRU eviction loses the snapshots that the next turn
  needs, and recompute rises to 20–40% at 64 slots and 60–79% at 16. Retention by value
  keeps it at 6–8% and 10–20%. This agrees with the synthetic comparison and with the
  pool-pressure reports in SGLang #33713 and #42443.
- Marconi-style admission depends on the reply reappearing in the next prompt. When it
  does not, it keeps almost nothing reusable. This is the same effect as on synthetic
  traces, and stronger here because nearly every request is an append.
- On these traces `boundary_value` sees only prompt and reply ends; its message-boundary
  placement is not exercised. Adding the demand point (`demand=true`) helps slightly at
  large pools and has mixed effects at 16 slots.

## Limits

- Simulated: sequential requests, unlimited KV, no time model. The pool pressure comes
  from interleaving 40 sessions with an assumed 60 s stagger; a server with a different
  arrival mix sees a different pressure.
- Whether replies round-trip depends on the served model's chat template; both cases are
  shown and neither is known to hold for these traces.
- One corpus of one agent. Subsets were taken in file order, not sampled.

## Reproduce

```bash
results/2026-10-15-sim-policy-public-traces/run.sh
```

Downloads the traces (about 630 MB) at the pinned commit into a temporary directory.

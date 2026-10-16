# What drives value-based retention's gain: an ablation (simulated)

**Result (simulated): on the recorded agent traces, `boundary_value`'s advantage under
state-pool pressure comes from value-based admission and eviction, not from where it
places snapshots.** With placement held fixed, replacing LRU eviction with value-based
eviction cuts recompute by 12–26% at 64 slots and 39–43% at 16 slots for prompt-end
placement. Reply-end snapshots help when the pool is unlimited (by up to 0.27 points).
At 128 slots and below they take slots from prompt-end snapshots and raise recompute. Configuration: [config.yaml](config.yaml).

## Setup

The traces, subsets, and simulator settings are the same as in the
[policy comparison](../2026-10-15-sim-policy-public-traces/summary.md): 120 kv-cache-tester
sessions in three subsets of 40, sequential requests, unlimited KV, and both reply
round-trip assumptions. On average 14 to 16 sessions are live at a time, so 64 slots is
about 4 per live session ([pool sweep](../2026-10-15-sim-state-pool-pressure/summary.md)).

| Policy | Placement | Eviction |
| --- | --- | --- |
| `prompt_end_anchor` | prompt end | least recently used |
| `boundary_value:eviction=lru` | prompt end and reply end | least recently used |
| `boundary_value:reply_end=false` | prompt end | by value; a new snapshot worth less than every resident one is not admitted |
| `boundary_value` | prompt end and reply end | by value, as above |

The traces record no message boundaries, so the `boundary_value` variants place at
prompt ends (and reply ends) only. `eviction=lru` is new in this change. It keeps the
placement, evicts the least recently used snapshot, and admits every new one.

## Results

Recomputed share of prompt tokens, subsets 1 / 2 / 3, replies replayed in the next
prompt. Simulated. With replies not replayed, the prompt-end-only policies give the same
numbers, because their snapshots never lie past the prompt. All cases are in
[raw/summary.json](raw/summary.json).

| Slots | `sglang_current` | `prompt_end_anchor` | `boundary_value:eviction=lru` | `boundary_value:reply_end=false` | `boundary_value` | Offline reference |
| --- | --- | --- | --- | --- | --- | --- |
| Unlimited | 3.92 / 4.86 / 3.76 | 4.97 / 5.28 / 4.78 | 4.70 / 5.08 / 4.57 | 4.97 / 5.28 / 4.78 | 4.70 / 5.08 / 4.57 | 3.75 / 4.70 / 3.66 |
| 256 | 4.95 / 8.61 / 7.29 | 5.60 / 6.24 / 5.90 | 5.70 / 7.23 / 6.73 | 5.50 / 5.54 / 5.29 | 5.43 / 5.81 / 5.44 | 3.93 / 4.74 / 4.10 |
| 128 | 7.53 / 17.86 / 22.12 | 6.09 / 7.60 / 7.09 | 6.75 / 9.33 / 8.78 | 5.76 / 6.22 / 5.98 | 5.88 / 6.61 / 6.45 | 4.03 / 4.89 / 4.10 |
| 64 | 21.65 / 39.98 / 36.62 | 7.34 / 9.89 / 9.06 | 9.63 / 12.45 / 12.20 | 6.44 / 7.29 / 6.93 | 6.81 / 8.28 / 7.91 | 4.43 / 5.04 / 4.23 |
| 32 | 42.99 / 66.22 / 51.82 | 10.71 / 14.24 / 13.36 | 14.45 / 19.93 / 20.71 | 7.48 / 9.14 / 8.68 | 8.34 / 11.31 / 10.93 | 4.74 / 5.49 / 4.45 |
| 16 | 62.03 / 78.90 / 70.35 | 17.87 / 23.30 / 23.16 | 26.31 / 31.42 / 30.79 | 10.15 / 13.82 / 14.06 | 12.39 / 20.01 / 19.89 | 5.55 / 6.72 / 5.26 |

Cut from value-based eviction at fixed placement, prompt ends only
(`prompt_end_anchor` → `boundary_value:reply_end=false`), subsets 1 / 2 / 3:

| Slots | 256 | 128 | 64 | 32 | 16 |
| --- | --- | --- | --- | --- | --- |
| Cut | 2% / 11% / 10% | 6% / 18% / 16% | 12% / 26% / 24% | 30% / 36% / 35% | 43% / 41% / 39% |

## Reading

- **Fewer snapshots first.** Under pressure, most of the gap between
  `sglang_current` and the other policies comes from how many snapshots each writes.
  `sglang_current` writes about 4.7 per request at 64 slots in subset 1, and
  `prompt_end_anchor` writes one. Pool churn evicts the snapshots the next turn needs.
- **Then value-based eviction.** At the same placement, ranking by value instead of
  recency removes a further 12–26% of recompute at 64 slots and about 40% at 16 slots.
- **Reply-end snapshots cost more than they save under pressure.** When replies
  round-trip, a reply-end snapshot saves the reply's tokens on the next turn. With an
  unlimited pool that is worth up to 0.27 points. At 256 slots the effect is mixed. At
  128 slots and below, each one takes a slot that a prompt-end snapshot would have used,
  and recompute rises in every case, by about 6 points at 16 slots in two of the three
  subsets.
- The reply-end setting was not chosen on these traces: both settings existed before
  this run and are reported as found. A policy for engines would still need to be
  checked on other traces before relying on this.

## Limits

- Simulated, with unlimited KV and sequential requests; the pool pressure comes from 40
  interleaved sessions with an assumed 60 s stagger.
- One corpus of one agent. Message-boundary placement is not exercised.
- `eviction=lru` admits every snapshot. The admission check in `boundary_value` and its
  eviction order are not separated here.

## Reproduce

```bash
results/2026-10-16-sim-retention-ablation/run.sh
```

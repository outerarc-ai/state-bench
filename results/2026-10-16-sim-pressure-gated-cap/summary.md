# Pressure-gated per-path cap (simulated)

**Result (simulated): applying a per-path cap of 2 only while the state pool is at least
80–95% full matches the uncapped engine when the pool has room, and the fixed cap
everywhere else.** When the pool never fills, recompute is the same as with no cap
(3.8–5.1%). At 64 slots it is 4.8–7.6%, against 19.8–40.0% with no cap and 4.8–7.5% with
a fixed cap of 2. The threshold makes little difference: every finite pool in these runs
reaches it, and from then on the gated and fixed caps evict alike. Configuration: [config.yaml](config.yaml).

## Setup

- The same traces, subsets, and simulator settings as the
  [per-path cap sweep](../2026-10-16-sim-path-cap/summary.md), whose fixed-cap,
  no-cap, and reference runs are the comparison here.
- `sglang_current` with a cap of 1 or 2, applied only while the pool holds at least
  80% or 95% of its slots (`--path-cap-min-usage`). This is not an SGLang option. It
  models one way an engine could tie the cap to pool pressure.

## Results

Recomputed share of prompt tokens over the six cases (3 subsets × 2 round-trip
assumptions). The gated cap is shown at the 95% threshold; the 80% threshold differs by
at most 0.2 points. Per-case values: [raw/summary.json](raw/summary.json).

| Slots | No cap | Fixed cap 2 | Gated cap 2 | Gated cap 1 | Value-based, prompt ends | Offline reference |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 60.0–78.9% | 13.2–20.2% | 13.2–20.2% | 11.9–17.8% | 10.1–14.1% | 4.3–6.7% |
| 32 | 40.3–66.2% | 5.7–10.5% | 5.7–10.5% | 6.3–10.5% | 7.5–9.1% | 3.8–5.5% |
| 64 | 19.8–40.0% | 4.8–7.5% | 4.8–7.6% | 5.4–8.0% | 6.4–7.3% | 3.8–5.0% |
| 128 | 6.4–22.1% | 4.4–6.8% | 4.5–6.7% | 4.7–7.3% | 5.8–6.2% | 3.8–4.9% |
| 256 | 5.0–8.6% | 4.3–6.6% | 4.4–6.6% | 4.5–7.1% | 5.3–5.5% | 3.8–4.9% |
| Unlimited | 3.8–5.1% | 4.3–6.6% | 3.8–5.1% | 3.8–5.1% | 4.8–5.3% | 3.7–4.9% |

Gated cap 2 compared case by case (percentage points; negative is lower recompute):

| Slots | 16 | 32 | 64 | 128 | 256 | Unlimited |
| --- | --- | --- | --- | --- | --- | --- |
| Against no cap | −59.7 to −45.2 | −57.2 to −32.5 | −33.7 to −12.8 | −17.6 to +0.3 | −3.6 to +1.6 | 0 |
| Against value-based retention | +3.0 to +6.3 | −1.8 to +1.3 | −1.6 to +0.6 | −1.5 to +1.0 | −1.1 to +1.1 | −1.1 to −0.2 |

## Reading

- **The gate removes the cap's cost only when the pool has room.** With an unlimited
  pool the cap never applies. Every finite pool in these runs fills, and the gated cap then
  behaves like the fixed one.
- **A cap of 2 is a reasonable single setting.** It is within about 1.6 points of the
  best fixed cap in each case from 32 to 256 slots, and it never costs anything when the
  pool has room. At 16 slots a cap of 1 is better, and value-based retention better still.
- **A full but large pool is the remaining weak spot.** At 256 slots, when replies are
  not replayed, a cap of 2 can recompute up to 1.6 points more than no cap. It evicts
  shallow snapshots that rewrites and forks would have reused, even though only a few
  slots are needed. Tying the cap to how far the pool is over budget, rather than to a
  fixed threshold, would address this; it has not been simulated.

## Limits

- Simulated, with unlimited KV and sequential requests; the pool pressure comes from 40
  interleaved sessions with an assumed 60 s stagger. Real pools also hold running
  requests' slots, which shifts when the threshold is crossed.
- Fork points are recognized only at block boundaries (see
  [SIMULATOR.md](../../docs/SIMULATOR.md)).
- One corpus of one agent.

## Reproduce

```bash
results/2026-10-16-sim-pressure-gated-cap/run.sh
```

It needs `results/2026-10-16-sim-path-cap/raw` for the comparison, downloads the traces
(about 630 MB) into a temporary directory, and takes about 40 minutes on two cores.

# SGLang's per-path snapshot cap under state-pool pressure (simulated)

**Result (simulated): on the recorded agent traces, SGLang's existing per-path cap
(`--mamba-max-states-per-path`, unlimited by default) recovers most of the recompute that
LRU retention loses when the state pool is short. No single cap value suits every pool
size.** At 64 slots (about 4 per live session), a cap of 2 lowers `sglang_current` from
19.8–40.0% to 4.8–7.5% of prompt tokens. That is as low as value-based retention
(6.4–7.3%) or lower in four of six cases. With an unlimited pool, the same cap raises
recompute from 3.8–5.1% to 4.3–6.6%. At 16 slots, the best cap still recomputes 14–25%
more than value-based retention. Configuration: [config.yaml](config.yaml).

## Setup

- **Traces.** The 120 kv-cache-tester sessions in three subsets of 40, as in the
  [policy comparison](../2026-10-15-sim-policy-public-traces/summary.md), with both reply
  round-trip assumptions and the same simulator settings.
- **The cap.** `sglang_current` runs with caps of 1, 2, 4, and 8 snapshots per path, and
  without one. The cap models SGLang `c4870d98`: after each new snapshot, the shallowest
  snapshots on its root-to-tail path beyond the cap are evicted. The new snapshot, fork
  points, and the snapshot a running request resumed from are kept
  ([SIMULATOR.md](../../docs/SIMULATOR.md)).
- **References.** `boundary_value:reply_end=false` (prompt ends, value-based admission
  and eviction; the best variant in the [retention
  ablation](../2026-10-16-sim-retention-ablation/summary.md)), `prompt_end_anchor`, and
  `offline_oracle` run without a cap.

## Results

Recomputed share of prompt tokens over the six cases (3 subsets × 2 round-trip
assumptions), simulated. Per-case values are in [raw/summary.json](raw/summary.json).

| Slots | No cap | Cap 1 | Cap 2 | Cap 4 | Cap 8 | Value-based, prompt ends | Offline reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 60.0–78.9% | 11.9–17.8% | 13.2–20.2% | 24.7–40.3% | 42.7–61.9% | 10.1–14.1% | 4.3–6.7% |
| 32 | 40.3–66.2% | 6.3–10.5% | 5.7–10.5% | 10.5–16.5% | 19.1–30.3% | 7.5–9.1% | 3.8–5.5% |
| 64 | 19.8–40.0% | 5.4–7.8% | 4.8–7.5% | 5.1–9.2% | 7.5–12.5% | 6.4–7.3% | 3.8–5.0% |
| 128 | 6.4–22.1% | 5.0–7.4% | 4.4–6.8% | 4.5–6.6% | 4.9–8.4% | 5.8–6.2% | 3.8–4.9% |
| 256 | 5.0–8.6% | 4.8–7.2% | 4.3–6.6% | 4.3–5.7% | 4.3–6.3% | 5.3–5.5% | 3.8–4.9% |
| Unlimited | 3.8–5.1% | 4.8–7.2% | 4.3–6.6% | 4.3–5.8% | 3.9–5.6% | 4.8–5.3% | 3.7–4.9% |

Value-based retention against the lowest cap setting in each case. A positive number
means value-based retention recomputes less. Choosing the cap per case is optimistic for
the cap, since a deployment cannot know the best value in advance.

| Slots | 16 | 32 | 64 | 128 | 256 | Unlimited |
| --- | --- | --- | --- | --- | --- | --- |
| Range over six cases | +14% to +25% | −31% to +13% | −34% to +2% | −34% to +1% | −25% to +4% | −27% to −4% |

## Reading

- **The cap stops the churn.** Without it, `sglang_current` writes about 4 to 18
  snapshots per request at 16 to 64 slots. Each write evicts something under LRU, often
  a snapshot the next turn needs. With a cap of 1 or 2, it writes 2 to 6 per request,
  because each path keeps only its deepest snapshots.
- **The right cap depends on the pool.** The best value moves from 1 at 16 slots, to 2
  at 32 to 64, to 2 or 4 at 128, to 4 or more at 256, to no cap when the pool is unlimited. A small
  cap set for a short pool costs up to 3.0 points when the pool is ample, mostly when
  replies do not round-trip: it evicts shallow snapshots that rewrites and forks would
  have used.
- **Value-based retention adds little where the cap works.** From 32 to 256 slots, a
  well-chosen cap matches or beats it in most cases. It is clearly ahead only when the
  pool holds about one slot per live session.
- An engine-side option that follows pool pressure (no cap when the pool has room,
  tighter as it fills) would combine the low-pressure and high-pressure behavior. It has
  not been simulated here.

## Limits

- Simulated, with unlimited KV and sequential requests. The pool pressure comes from 40
  interleaved sessions with an assumed 60 s stagger.
- Fork points are recognized only at block boundaries. SGLang also sees forks inside a
  block, so it may keep a few snapshots that the simulator evicts.
- Value-based retention was not combined with a cap here.
- One corpus of one agent.

## Reproduce

```bash
results/2026-10-16-sim-path-cap/run.sh
```

It downloads the traces (about 630 MB) at the pinned commit into a temporary directory
and takes about 75 minutes on two cores.

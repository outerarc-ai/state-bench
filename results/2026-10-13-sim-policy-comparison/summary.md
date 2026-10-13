# Snapshot policy comparison on synthetic agent traces

**Result: under state-pool pressure, retention decides most of the recomputation, and
the gap to the offline reference stays large for every online policy.** In the six main
cases (shared system prompt, replies round-trip), every policy except the prompt-end
anchor is within about 0.5 points of the offline reference when the pool is unlimited. At 64
slots for 64 sessions, in the seven cases where replies round-trip, SGLang's current
placement recomputes 16.5% to 39.6% of prompt tokens while the reference recomputes
6.6% to 7.6%.
`boundary_value` has the lowest recomputation among online policies at 64 slots in all
ten cases, at the cost of about twice as many snapshot writes as Marconi-style
admission.

All numbers are **simulated** on synthetic traces whose parameters are assumptions.
They compare policies under the simulator's model; they are not measurements of any
engine and say nothing about latency. Configuration: [config.yaml](config.yaml).
Regenerate with [run.sh](run.sh).

## Results

Recomputed share of prompt tokens at 64 state slots (64 sessions):

| Case | sglang_current | prompt_end_anchor | marconi_admission | pin_recent | boundary_value | offline_oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed-11 | 16.5% | 15.3% | 11.7% | 12.4% | **9.3%** | 7.2% |
| mixed-12 | 23.8% | 17.8% | 14.3% | 17.4% | **11.2%** | 7.1% |
| mixed-13 | 30.7% | 20.6% | 16.5% | 23.7% | **13.3%** | 7.4% |
| fork-11 | 18.8% | 15.6% | 11.8% | 12.9% | **9.2%** | 6.6% |
| fork-12 | 24.4% | 17.0% | 13.5% | 16.4% | **10.2%** | 6.7% |
| fork-13 | 39.6% | 22.1% | 18.7% | 27.6% | **14.4%** | 6.7% |
| mixed-11, no shared system prompt | 20.7% | 15.3% | 14.5% | 14.7% | **11.8%** | 7.6% |
| mixed-11, replies do not round-trip | 13.4% | 15.3% | 70.0% | 12.2% | **10.6%** | 8.3% |
| mixed-21, replies do not round-trip | 12.7% | 14.7% | 67.8% | 11.6% | **10.4%** | 7.7% |
| fork-22, replies do not round-trip | 14.2% | 14.2% | 71.1% | 11.3% | **9.9%** | 7.5% |

Unlimited, 256, and 16 slots, and snapshot writes, are in `raw/`. Selected values for
mixed-11:

| Policy | Unlimited | 256 | 64 | 16 | Writes (unlimited) |
| --- | ---: | ---: | ---: | ---: | ---: |
| sglang_current | 7.6% | 8.5% | 16.5% | 57.5% | 5,722 |
| prompt_end_anchor | 10.1% | 10.5% | 15.3% | 39.5% | 3,258 |
| marconi_admission | 7.7% | 7.8% | 11.7% | 38.7% | 2,986 |
| pin_recent | 7.6% | 8.4% | 12.4% | 52.8% | 5,722 |
| boundary_value | 7.3% | 7.6% | 9.3% | 28.5% | 6,195 |
| offline_oracle | 7.2% | 7.2% | 7.2% | 9.1% | 3,120 |

## Findings

1. **Placement matters little when the pool is ample; retention matters a lot when it
   is not.** With an unlimited pool, in the six main cases, the policies other than the
   prompt-end anchor are within about 0.5 points of the offline reference. At 64 slots the
   online policies range from 9.2% to 39.6% in the same cases, while the offline
   reference stays at 6.6% to 7.4%.
2. **SGLang's one-snapshot-per-chunk placement fills a small pool.** It writes a
   snapshot for every prefill chunk, and under LRU those evict the snapshots that the
   next turn needs. Its recomputation at 64 slots is the highest of all policies in
   seven of ten cases.
3. **Keeping only the last state of the whole sequence fails when replies do not
   round-trip.** `marconi_admission` stores the last grid point of everything a
   request computed, which lies inside the reply. When the next request does not
   contain the reply token for token, nothing is reusable and recomputation rises to
   67.8% to 71.1% at every pool size. This is the failure mode reported in vLLM
   [#45238](https://github.com/vllm-project/vllm/issues/45238) (open as of
   2026-10-12). With round-tripping replies the same policy is among the best.
4. **The prompt-end anchor alone misses branch points.** Without demand snapshots it
   never stores the shared system prompt or a fork point, so it is the worst policy
   with an unlimited pool (9.0% to 10.3%, vs 6.6% to 8.5% for the others, excluding
   `marconi_admission` when replies do not round-trip). Its few writes keep it among
   the better policies at 16 slots.
5. **`boundary_value` is ahead of the other online policies at 64 slots in every case,
   but it relies on two favorable assumptions.** The synthetic generator places every
   branch point on a message boundary, which is exactly where this policy stores
   snapshots, and the policy assumes several snapshots can be captured per prefill
   chunk. It writes about twice as many snapshots as `marconi_admission`. Keeping both
   the prompt end and the reply end of the latest turn makes it robust to replies that
   do not round-trip, and costs it at 16 slots with round-tripping replies (28.5% to
   46.4%, still below the other online policies).

## Limits

- Synthetic traces only. Real agent traces may branch inside messages, reuse system
  prompts differently, and round-trip replies at an unknown rate.
- Requests run one at a time; in-flight state slots and KV capacity are not modeled, so
  pool pressure appears later than in a real server.
- Recomputed tokens are not latency. Request-level effects need a cost model and
  measurement under the [measurement protocol](../../docs/MEASUREMENT.md).
- `boundary_value` was designed on seeds 1 to 3 and revised once after this
  directory's first run (see `design_history` in `config.yaml`).

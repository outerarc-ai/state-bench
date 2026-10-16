# State-pool pressure on recorded agent traces (simulated)

**Result (simulated): on 120 recorded Claude Code sessions, the choice of snapshot
retention starts to matter when the state pool holds fewer than about 10 slots per live
session.** At
about 4 slots per live session, SGLang's current LRU retention recomputes 20–40% of
prompt tokens and value-based retention (`boundary_value`) 6.0–8.3%. With an unlimited
pool, the best existing policy is within 0.3 points of the offline reference. **Under SGLang's
default memory split, a session holding its whole context in KV gets 11 to 63 state
slots at the median context of 109k tokens (six hybrid models), but only 1 to 6 at 10k
tokens.** Configuration: [config.yaml](config.yaml).

This extends [the four-size comparison](../2026-10-15-sim-policy-public-traces/summary.md)
to twelve pool sizes on the same traces and settings. It expresses the pool as slots per
live session, so it can be compared with deployments.

## Method

- **Simulation.** Same traces, subsets, and simulator settings as the earlier
  comparison: sequential requests, unlimited KV, `chunked_prefill_size` 8192 with the
  hashed first chunk cut, and both reply round-trip assumptions. Pools range from 16 to
  512 slots, plus unlimited.
- **Live sessions.** A session is live from its first request's arrival to its last.
  Averaged over request arrivals, each 40-session subset has 14.1 to 15.8 live sessions
  ([live_sessions.py](live_sessions.py)). Slots per live session is the pool size divided
  by that mean.
- **Deployment.** When `--max-mamba-cache-size` is not set, SGLang gives recurrent state
  and KV the memory left after weights in the ratio `mamba_full_memory_ratio` : 1,
  default 0.9 (SGLang `c4870d98`, `mem_cache/kv_cache_configurator.py:3016-3028`). If
  every live session keeps its whole context in KV, each session's share of state slots
  does not depend on GPU memory:
  `slots per session = 0.9 × context × KV bytes per token ÷ state bytes per slot`.
  Sizes are per GPU, from the published configs ([models.json](models.json),
  [deployment.py](deployment.py)).

## Results

### Recompute by pool size

Recomputed share of prompt tokens over 3 subsets × 2 round-trip assumptions (6 cases),
simulated. "Best existing" is the lowest of `sglang_current`, `prompt_end_anchor`,
`marconi_admission`, and `pin_recent` in each case. `boundary_value` is the lower of its
two settings. Data: [raw/summary.json](raw/summary.json).

| Pool (slots) | Slots per live session | `sglang_current` | Best existing | `boundary_value` | Offline reference | `boundary_value` ≥ 10% below best existing |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 1.0–1.1 | 60.0–78.9% | 17.9–23.3% | 10.0–20.0% | 4.3–6.7% | 6 of 6 |
| 32 | 2.0–2.3 | 40.3–66.2% | 10.1–14.2% | 7.3–11.3% | 3.8–5.5% | 6 of 6 |
| 64 | 4.1–4.6 | 19.8–40.0% | 6.6–9.9% | 6.0–8.3% | 3.8–5.0% | 5 of 6 |
| 128 | 8.1–9.1 | 6.4–22.1% | 5.5–7.6% | 5.3–6.5% | 3.8–4.9% | 4 of 6 |
| 192 | 12.2–13.6 | 5.3–10.4% | 5.2–7.2% | 5.1–6.1% | 3.8–4.9% | 3 of 6 |
| 256 | 16.3–18.2 | 5.0–8.6% | 5.0–6.2% | 5.0–5.8% | 3.8–4.9% | 2 of 6 |
| 512 | 32.5–36.4 | 4.5–6.5% | 4.5–5.9% | 4.6–5.3% | 3.7–4.9% | 2 of 6 |
| Unlimited | — | 3.8–5.1% | 3.8–5.1% | 4.3–5.2% | 3.7–4.9% | 0 of 6 |

Ranges span the six cases. Pools of 24, 48, 96, and 384 slots fall between their
neighbors and are in the raw data.

### State slots per session under SGLang's default split

Derived from published model configs and SGLang's default ratio. The slot size is also
shown in tokens of KV.

| Model (TP) | KV per token per GPU | State per slot per GPU | Slot in KV tokens | Slots per session at 10k / 30k / 109k / 297k context |
| --- | --- | --- | --- | --- |
| NVIDIA-Nemotron-Nano-9B-v2 (1) | 16 KiB | 136.9 MiB | 8,762 | 1.0 / 3.1 / 11.2 / 30.5 |
| Qwen3.5-9B (1) | 32 KiB | 49.1 MiB | 1,572 | 5.7 / 17.2 / 62.5 / 170.2 |
| Qwen3.5-27B (1) | 64 KiB | 146.8 MiB | 2,349 | 3.8 / 11.5 / 41.8 / 113.9 |
| Qwen3.5-35B-A3B (1) | 20 KiB | 61.4 MiB | 3,144 | 2.9 / 8.6 / 31.2 / 85.1 |
| Qwen3.5-122B-A10B (2) | 12 KiB | 73.3 MiB | 6,252 | 1.4 / 4.3 / 15.7 / 42.8 |
| Qwen3.5-397B-A17B (8) | 15 KiB | 22.9 MiB | 1,563 | 5.8 / 17.3 / 62.8 / 171.1 |

Each running request also holds 5 slots: 3, plus 2 tracking slots with the extra buffer
and the overlap scheduler. Full output: [raw/deployment.json](raw/deployment.json).

## Reading

- **Where the policy matters.** Below about 10 slots per live session, SGLang's LRU
  retention loses the snapshots the next turn needs, and recompute grows quickly: 6–22%
  at 8–9 slots, 20–40% at about 4. Above about 17, every online policy is within 4 points of
  the reference, except Marconi-style admission when replies do not round-trip (about
  96% at every size).
- **Where deployments sit.** Long agent contexts put each KV-resident session at 11 or
  more slots under the default split, near the edge of that region for
  NVIDIA-Nemotron-Nano-9B-v2 and well above it for the Qwen3.5 models. At 10k tokens all
  six models sit inside it, and at 30k three of the six.
- **What can push long-context sessions into it.** Sessions whose KV has moved to a host
  tier still need their state snapshots, but share a device pool sized for the
  KV-resident ones; the 5 slots held by each running request also come out of the pool.
- **Where `boundary_value`'s gain comes from.** On these traces it places snapshots at
  prompt and reply ends only, as `prompt_end_anchor` does, so its gain comes from
  value-based admission and eviction rather than placement.

## Limits

- Simulated, with unlimited KV. In an engine, KV eviction also drops prefixes, which
  changes which snapshots are useful.
- A live session in the simulation keeps its whole context. In the deployment table, the
  per-session share assumes the KV pool is full of live sessions' contexts; with fewer
  sessions there are more slots per session.
- The live-session count depends on the assumed 60 s stagger between session starts.
- Sizes are from published configs and SGLang's default split. Other settings
  (`--max-mamba-cache-size`, a different ratio, FP8 KV) change them.

## Reproduce

```bash
results/2026-10-15-sim-state-pool-pressure/run.sh
```

It downloads the traces (about 630 MB) at the pinned commit into a temporary directory
and takes about 30 minutes on two cores.

# Restore versus recompute on MI300X: first estimate

**Result (estimate, not a measurement): for NVIDIA-Nemotron-Nano-9B-v2 on one MI300X,
restoring a chain's KV cache and Mamba state beats evicting and recomputing it once the
prefix exceeds about 90–150 tokens from host memory, or about 530–910 tokens from local
NVMe.** The crossover is set almost entirely by the fixed 137 MiB state snapshot. At the
context lengths seen at idle gaps in public agent traces (median 109k tokens), recompute
takes 3.7–6.2 s; restoring from host memory adds about 40 ms to the time to first token,
and from NVMe about 280 ms. Inputs and their sources: [inputs.json](inputs.json); method:
[estimate.py](estimate.py); output: [raw/estimate.json](raw/estimate.json).

## Model

Saving to a tier and restoring wins when

```
(s_kv * L + S_state) / B_tier + t_fixed < T_prefill(L)
T_prefill(L) = (2 * P * L + A * L^2) / (peak * MFU)
```

with `A * L^2` the causal attention FLOPs. Time to first token after waking with `new`
appended tokens is the restore time (zero when kept in HBM) plus the incremental prefill
of `new` tokens over `L` cached ones; recomputing prefills `L + new` from scratch.

| Input | Value | Source |
| --- | --- | --- |
| Parameters `P` | 8.84 B | derived from measured weight memory (gfx1151) |
| KV per token `s_kv` | 16 KiB | measured KV allocation (gfx1151) |
| State snapshot `S_state` | 136.9 MiB | derived from the model config |
| Attention layers × query heads × head dim | 4 × 40 × 128 | spec: config.json, checked 2026-10-17 ([prefill curve](../2026-10-17-strix-halo-prefill-curve/summary.md)) |
| Peak dense bf16 | 1,307 TFLOPS | spec (MI300X) |
| Prefill MFU | 30% and 50% | assumption |
| Host memory → HBM | 50 GB/s, 1 ms fixed | assumption (PCIe Gen5 x16 spec: 64 GB/s per direction) |
| Local NVMe → HBM | 7 GB/s, 2 ms fixed | assumption (one Gen4 drive per GPU) |
| Contexts and new tokens | 10k, 109k, 297k; 666 and 3,462 new | recorded trace properties ([idle gaps](../2026-10-15-public-trace-idle-gaps/summary.md)) |

## Results

All values are estimates.

Crossover prefix length `L*` (restore is faster above it):

| Tier | MFU 30% | MFU 50% |
| --- | --- | --- |
| Host memory | 87 tokens | 145 tokens |
| Local NVMe | 526 tokens | 909 tokens |

Time to first token after waking, 666 new tokens (median append), MFU 30% / 50%:

| Context | Held | Keep in HBM | Restore from host | Restore from NVMe | Recompute |
| --- | --- | --- | --- | --- | --- |
| 10k | 0.31 GB | 32 / 19 ms | 39 / 26 ms | 77 / 65 ms | 493 / 296 ms |
| 109k (median at a gap) | 1.93 GB | 45 / 27 ms | 85 / 67 ms | 323 / 305 ms | 6.2 / 3.7 s |
| 297k (p90 at a gap) | 5.01 GB | 71 / 43 ms | 173 / 144 ms | 790 / 761 ms | 22.7 / 13.6 s |

With 3,462 new tokens (p90 append) every option except recompute grows by the same
incremental prefill, 0.1–0.3 s; see the raw output.

## Reading

- For this hybrid model, evicting and recomputing is the slowest option at every context
  length in the traces, by one to two orders of magnitude. The useful question is which
  tier to keep a chain in, and when to bring it back, not whether to keep it.
- Even from NVMe, restoring a median context takes about 0.3 s, well under the median
  idle gap of 10 s ([idle gaps](../2026-10-15-public-trace-idle-gaps/summary.md)). If the
  engine starts the restore before the next request arrives, for example from a hint of
  the expected return time, the wake-up can cost about as much as keeping the chain in HBM.
- The state snapshot is a fixed cost of about 9k tokens' worth of KV. It dominates the
  restore time for short prefixes and keeps the crossover above zero.

## Limits

- A single request on an otherwise idle GPU. Under load, recompute also takes compute
  from other requests, and restore traffic shares PCIe and storage bandwidth; neither is
  modeled.
- Restores are modeled as one copy before compute starts; a layer-wise restore that
  overlaps with compute would be faster.
- MFU, tier bandwidths, and fixed costs are assumptions until measured on MI300X. The
  attention shape is taken from the published config and was not read from the model
  files used here.
- One model. Dense-attention models have no state snapshot and much more KV per token,
  which moves the crossover and the restore times.

## Reproduce

```bash
results/2026-10-15-restore-vs-recompute-estimate/run.sh
```

## Follow-up (2026-10-17)

The model's attention shape matches the assumption above. A prefill curve on gfx1151
([2026-10-17-strix-halo-prefill-curve](../2026-10-17-strix-halo-prefill-curve/summary.md))
shows the attention term about 1.25 times as expensive, relative to the rest of the model,
as its FLOPs imply. If MI300X behaved the same, recompute at 109k tokens would take about 5%
longer than estimated here, which favors restoring further. The estimate itself is
unchanged until the curve is measured on MI300X.

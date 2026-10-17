# Prefill time against prompt length on gfx1151

**Result: on a shared development GPU, prefill time for NVIDIA-Nemotron-Nano-9B-v2 grows
from 0.92 ms per token at 1k tokens to 1.17 ms per token at 64k tokens (measured), a factor
of 1.28. A FLOP count of the same model predicts 1.15, so the attention term is about 1.25
times as expensive, relative to the rest, as FLOPs alone imply. The model's attention shape
matches the restore-versus-recompute estimate's assumption.** These times are from gfx1151
and say nothing about MI300X; they check the form of the estimate's prefill model.
Configuration: [config.yaml](config.yaml); script: [run.sh](run.sh); analysis:
[analyze.py](analyze.py) → [data.json](data.json); log: [raw/round17-partA.log](raw/round17-partA.log).

| Prompt tokens | Median time (s) | Runs (s) | µs per token | Fit (s) |
| ---: | ---: | --- | ---: | ---: |
| 1,024 | 0.941 | 0.922, 0.941, 0.954 | 919 | 0.977 |
| 2,048 | 1.930 | 1.962, 1.913, 1.930 | 942 | 1.996 |
| 4,096 | 3.990 | 3.940, 3.990, 4.027 | 974 | 4.052 |
| 8,192 | 8.406 | 8.240, 8.409, 8.406 | 1,026 | 8.235 |
| 16,384 | 16.976 | 17.033, 16.829, 16.976 | 1,036 | 16.890 |
| 32,768 | 35.237 | 35.088, 35.237, 35.416 | 1,075 | 35.352 |
| 65,536 | 76.907 | 76.907, 76.455, 77.234 | 1,174 | 76.886 |

Times are measured: end-to-end request time with one output token, prefix cache flushed
before each request, three repetitions in alternating length order. The fit
`t = c + a·L + b·L²` is derived from the medians: a = 986 µs per token, b = 2.86 ps per
token², c = −0.04 s.

## Comparison with the FLOP model

The estimate prefills `L` tokens in time proportional to `2·P·L + A·L²`, with `P` = 8.84 B
parameters (derived from measured weight memory) and `A = 2 × layers × heads × head_dim`.

- **Attention shape.** The model's `config.json` gives 40 query heads, 8 KV heads, head
  dimension 128, and a hybrid pattern of 27 Mamba, 4 attention, and 25 MLP layers (spec,
  printed in the log). The estimate assumed 4 × 40 × 128; it matches.
- **Quadratic share.** In the FLOP model the attention term is `A/(2P)` = 2.3 × 10⁻⁶ of
  the linear term per token of context; the fit gives `b/a` = 2.9 × 10⁻⁶ (derived). On this
  GPU and backend (Triton attention), attention runs at a lower FLOP rate than the rest of
  the model. Effective throughput is 17.4–19.3 TFLOPS by the FLOP model (derived), falling
  with length.
- **Effect on the estimate.** If MI300X showed the same ratio, recompute at the 109k-token
  median idle context would take about 5% longer than the estimate states, which favors
  restoring further. Whether it does needs the same curve on MI300X
  ([envs/mi300x/prefill_curve.py](../../envs/mi300x/prefill_curve.py)).

## Notes

- At server start, aiter's JIT build of `module_custom_all_reduce` fails on gfx1151
  (`hipcc` error in the log) and `amdsmi` is missing; the server starts and serves without
  them on one GPU.
- SGLang's shutdown ends its own process tree with SIGKILL, which the shell reports as
  `Killed` in the log; this is expected.
- A first attempt the same morning stopped at the first request: `/flush_cache` answers in
  plain text, and the client parsed it as JSON. The script now accepts either.

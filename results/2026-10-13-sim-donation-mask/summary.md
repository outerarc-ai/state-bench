# Donation-mask regression (SGLang #40907) in the simulator

**Result: the simulator reproduces the direction of the regression fixed by SGLang
[#40907](https://github.com/sgl-project/sglang/pull/40907), and its size depends on two
conditions.** The ratio approaches the reported magnitude when a prompt spans several
prefill chunks and the reply is not replayed token for token in the next request. When
the reply is replayed, the decode snapshot at completion hides most of the loss.

All numbers below are **simulated** on synthetic traces whose parameters are
assumptions. They show whether the simulator behaves like the engine; they are not
measurements of SGLang. Configuration: [config.yaml](config.yaml). Regenerate with
[run.sh](run.sh).

## Setup

- Traces: `synth` presets `append` (1,371 requests, median prompt 32,092 tokens) and
  `mixed` (3,530 requests, median prompt 19,536 tokens), 64 sessions, seed 1.
- `sglang_current` uses SGLang's prefix-relative donation mask, the non-DCP behavior after
  #40907. `sglang_absolute_mask` uses the absolute-grid mask, which SGLang applies with
  DCP and which applied without DCP between #39115 and #40907.
- `first_chunk_cut: hashed` cuts the first of several chunks at a per-request length, a
  stand-in for a shared prefill budget running out. Without it, every chunk starts on
  the grid and the two masks agree.
- State pool unlimited, so the difference comes from placement alone.

## Results

Recomputed prompt tokens, absolute vs relative mask, `first_chunk_cut: hashed`:

| Trace | Chunk size | Chunks per median prompt | Reply round-trips | Relative | Absolute | Ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| append | 2048 | ~16 | yes | 3.04M | 3.50M | 1.15x |
| append | 2048 | ~16 | no | 3.42M | 24.30M | 7.11x |
| append | 8192 | ~4 | yes | 3.01M | 3.11M | 1.04x |
| append | 8192 | ~4 | no | 3.38M | 5.24M | 1.55x |
| append | 16384 | ~2 | yes | 3.01M | 3.06M | 1.02x |
| append | 16384 | ~2 | no | 3.38M | 3.70M | 1.10x |
| mixed | 2048 | ~10 | yes | 7.31M | 8.49M | 1.16x |
| mixed | 2048 | ~10 | no | 8.19M | 38.48M | 4.70x |
| mixed | 8192 | ~2.4 | yes | 7.22M | 7.56M | 1.05x |
| mixed | 8192 | ~2.4 | no | 8.09M | 11.38M | 1.41x |
| mixed | 16384 | ~1.2 | yes | 7.22M | 7.35M | 1.02x |
| mixed | 16384 | ~1.2 | no | 8.09M | 8.75M | 1.08x |

With `first_chunk_cut: none`, all twelve configurations give identical results for the
two masks (ratio 1.00x; see `raw/`).

For reference, the #40907 author measured 1.82x (concurrency 20) and 1.83x
(concurrency 36) more computed prompt tokens with the absolute mask, on AgentX with a
median input of about 90k-100k tokens and `chunked_prefill_size` 16384, which is about
6 to 10 chunks per prompt. In the simulator, prompts of about 2.4 to 16 chunks without
reply round-trip give 1.41x to 7.11x, a range that contains the reported value.

## Findings

1. **Mechanism matches.** Once a chunk boundary leaves the absolute grid, no later chunk
   of that request donates a state, and because inserted KV stops at the last snapshot
   (R10), the KV for those chunks is not reused either.
2. **The decode snapshot masks the regression when replies round-trip.** At completion,
   SGLang inserts the last decode snapshot (R13), which lies on the absolute grid and
   covers the whole prompt. If the next request starts with the reply token for token,
   it resumes there and loses little. If the reply does not round-trip (for example,
   reasoning text removed by the chat template), the next request falls back to the
   prefill snapshots and loses everything after the first off-grid cut.
3. **How often replies round-trip is a workload property this repository cannot yet
   measure.** It decides whether this class of placement bug costs a few percent or
   most of the reuse. It is listed as an open question in
   [SGLang Mamba checkpoints](../../docs/SGLANG_MAMBA_CHECKPOINTS.md) and needs recorded
   traces or server-side counters to answer.

## Deviations from the upstream setup

- The upstream run used speculative decoding (MTP) and HiCache; the simulator models
  neither.
- In the upstream diagnostic, the first chunk of a request ended off the grid (at 4,253
  tokens) and no later chunk of that request donated a state. `first_chunk_cut: hashed`
  reproduces this pattern, but its distribution of cut lengths is an assumption.
- Requests run one at a time, so concurrency effects on the prefill budget are not
  modeled.

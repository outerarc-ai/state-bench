# Mamba + Unified Radix Cache KL test on gfx1151

**Result: all three multi-turn KL cases pass on gfx1151 with Triton backends and an
adequately sized state pool.** This is a functional check on a shared development
GPU; it says nothing about performance. Configuration: [config.yaml](config.yaml).

| Case | Avg KL (threshold 0.005) | Result |
| --- | --- | --- |
| `test_multiturn_logprobs_match` | 3.6e-4 | pass |
| `test_multiturn_prefill_cache_hit_branching` | 4.2e-4 | pass |
| `test_multiturn_decode_cache_hit_branching` | 3.9e-4 | pass |

All values measured. The first case comes from a run with 22 state slots; the other
two from a run with 64 slots, peak state-pool usage 56%.

## Findings

1. **The default attention backend aborts on gfx1151.** Without `--attention-backend`,
   SGLang selects aiter. Server init then fails with
   `HSA_STATUS_ERROR_EXCEPTION` in `paged_attention_ll4mi_QKV_mfma16_kernel`, which
   uses MFMA instructions that RDNA (gfx11xx) does not have. With Triton attention,
   linear-attention, and Mamba backends the server starts and serves correctly.
   This is specific to gfx1151 and does not indicate a problem on CDNA (gfx942).
2. **An aiter JIT build failure is non-fatal.** `module_custom_all_reduce` fails to
   compile (`__builtin_amdgcn_cvt_pk_fp8_f32` needs `fp8-conversion-insts`); aiter
   disables custom all-reduce and serving continues. Irrelevant at TP1.
3. **Cache-hit cases fail when the state pool is small, not because of a cache
   defect.** With less free memory at launch, auto-sizing gave
   `max_mamba_cache_size=22` with 5 state slots per request (`extra_buffer`), capping
   concurrency at 4. The third request of the first turn then reported
   `cached_tokens=0` (expected ≥1280 and ≥896). KV usage peaked at 16% while
   state-pool usage reached 73%: the state pool, not KV, was the binding limit.
   With 64 slots both cases pass.
4. **Long prompts produce sane output.** With chat templates, a 4,729-token prompt
   gives a correct summary with chunked prefill at 2,048 and at 16,384 tokens;
   greedy outputs diverge from token 4 but both are coherent, consistent with
   bf16 reduction-order differences. A 3,000-token plain continuation is
   token-identical across chunk sizes. Garbled text in the KL test log comes from
   the random-token suffixes the test appends.

## Deviations from the upstream test

- Model Qwen3.5-4B instead of Qwen3-Next-80B-A3B-Instruct-FP8; TP1 instead of TP4;
  `mem_fraction_static` 0.35 instead of 0.85 (shared GPU).
- Triton backends and `--max-mamba-cache-size 64` added (findings 1 and 3).
- Offline inputs instead of LongBench-v2; the GSM8K case was not run.
- `test_mamba_cache_kl.py` was not run: it was added upstream on 2026-10-11 with the
  fix it tests (SGLang #43567, merged), after this image was built.

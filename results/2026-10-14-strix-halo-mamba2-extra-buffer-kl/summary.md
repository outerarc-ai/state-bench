# Mamba2 extra_buffer KL test on gfx1151

**Result: both cases pass on gfx1151, after working around four issues that stop
Nemotron-H from starting on ROCm. Three of them are in SGLang's platform dispatch and are
present on current upstream main.** This is a functional check on a shared development
GPU; it says nothing about performance. Configuration: [config.yaml](config.yaml).

| Case | Avg KL | Threshold | Result |
| --- | --- | --- | --- |
| `test_input_output_logprobs_match_decode_cache_hit` | 4.57e-5 | 0.005 | pass |
| `test_input_output_logprobs_match_prefill_cache_hit` | 8.28e-5 | 0.012 | pass |

All values measured, 16 samples per case. Cases are listed in pytest's run order, which
is how the KL values map to them; the run took 581 s and peak state-pool usage was 67%
of 96 slots. Raw output: [raw/round15-summary.log](raw/round15-summary.log).

## Issues found on the way

Each one stopped the server at startup. Source references are to SGLang `9f5d871`
(the image's commit); the same code is on upstream main `684fea8` (2026-10-13). Issues 1
to 3 do not depend on gfx1151 and are expected to affect MI300X as well; that has not
been reproduced yet.

1. **Nemotron-H's default MoE backend pulls in FlashInfer on ROCm.**
   `arg_groups/model_overrides/nemotron_h.py:79-101` sets `moe_runner_backend` to
   `flashinfer_cutlass` for unquantized models on every platform other than sm100 (and
   modelopt FP4 on CUDA), even for a model without MoE layers.
   `model_executor/runner/flashinfer_autotune.py` gates warmup autotuning on
   `device == "cuda"` and a device capability major of at least 9; on ROCm these are
   `"cuda"` and the gfx major version, so autotuning runs and imports `flashinfer`, which
   ROCm images do not ship. Worked around with `--moe-runner-backend triton`.
2. **The Mamba2 mixer has no ROCm kernel binding.** `layers/attention/mamba/mamba.py`
   imports `causal_conv1d_fn` / `causal_conv1d_update` (and their `_triton` aliases) only
   under `is_cuda()`, `is_npu()`, and `is_xpu()`. `is_cuda()` is false on ROCm, so the
   names are undefined and decode graph capture fails with `NameError`. Worked around with
   a local patch that binds the Triton kernels on ROCm, as the XPU branch does
   ([local.patch](local.patch)). A 2025 fix for a Mamba2 crash on ROCm,
   [#11418](https://github.com/sgl-project/sglang/pull/11418), was closed without merging.
3. **`ReLU2` has no ROCm path.** `layers/activation.py` imports `relu2` only on CUDA, and
   `ReLU2` defines no `forward_hip`, so ROCm falls back to `forward_cuda` and fails with
   `NameError`. Worked around with a local patch that uses the native implementation.
4. **The Lean decode kernel exceeds gfx1151's shared memory.** On ROCm the Triton backend
   may select the Lean (work-centric) decode kernel, which its code describes as validated
   on MI300X and MI355X; on gfx1151 it raises `OutOfResources: shared memory`
   ([raw/round13-scheduler-exception.log](raw/round13-scheduler-exception.log)). This is
   specific to the development GPU. Worked around with `SGLANG_DISABLE_LEAN_ATTENTION=1`.

A search of SGLang issues and pull requests on 2026-10-14 found no open report of
issues 1 to 3; the search was not exhaustive.

## Deviations from the upstream test

- State pool of 96 slots with `--mem-fraction-static 0.39` instead of 256 slots with 0.8,
  to keep this server under 40 GiB on a host shared with other users. A first attempt with
  256 slots at 0.4 left no memory for the KV cache.
- Triton attention and Mamba backends; the default aiter attention aborts on gfx1151
  (see [the earlier functional check](../2026-10-13-strix-halo-mamba-radix-kl/summary.md)).
- The four workarounds above, including two temporary source patches.
- Offline inputs cut from the Python standard library instead of the test's default
  dataset, which the container cannot download.

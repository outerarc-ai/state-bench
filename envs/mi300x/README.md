# MI300X run scripts

Scripts that produce state-bench measurements on one AMD Instinct MI300X (gfx942) with
SGLang's ROCm image. They follow the [measurement protocol](../../docs/MEASUREMENT.md):
record the environment, check correctness before timing, run baseline and candidate as
alternating pairs on an idle device, and keep every raw record.

| Script | What it does |
| --- | --- |
| [`container.sh`](container.sh) | Starts (or reuses) the SGLang ROCm container with this repository mounted read-only at `/state-bench` and a work directory at `/work`, then runs one of the scripts below inside it |
| [`env_check.sh`](env_check.sh) | Records GPU, clocks, power, driver, ROCm, package versions, and SGLang commit; `--idle` prints `device idle` or the processes holding the GPU |
| [`nemotron_h_repro.sh`](nemotron_h_repro.sh) | Reproduces the three issues that stop Nemotron-H (Mamba2 hybrid) models from starting under SGLang on ROCm, then runs SGLang's Mamba2 `extra_buffer` KL test with test fixes applied. Self-contained |
| [`hybrid_cache_tests.sh`](hybrid_cache_tests.sh) | Single-GPU correctness checks of the hybrid prefix cache: SGLang's `test_mamba_cache_kl.py`, a one-GPU copy of `test_unified_radix_cache_kl_mamba.py`, and few-shot GSM8K with and without `--mamba-max-states-per-path` |
| [`paired_path_cap.sh`](paired_path_cap.sh) | Paired runs of `--mamba-max-states-per-path` against the default at several state-pool sizes, using `state-bench paired` |
| [`prefill_and_copy.sh`](prefill_and_copy.sh) | Prefill time against prompt length ([`prefill_curve.py`](prefill_curve.py)) and host-device copy bandwidth ([`copy_bandwidth.py`](copy_bandwidth.py)), the measured inputs of a restore-versus-recompute estimate |

## Requirements

- One MI300X with Docker access to `/dev/kfd` and `/dev/dri`.
- The image `rocm/sgl-dev:v0.5.21-rocm724-mi30x-20261014` (override with `IMAGE=`). Record
  its digest; `container.sh` prints it.
- Network access to Hugging Face and GitHub, and disk for the models: about 200 GB for all
  sessions below (assumption, from the published model sizes).
- No other process on the GPU while timing. `env_check.sh --idle` lists every process with
  an open KFD handle on the host; on a host whose other GPUs are in use, it reports busy and
  the idle check needs a narrower command.

## Sessions

Each session is a separate rental of a single GPU. Times are estimates (assumption) that
include model downloads and server start-up; they are not measurements.

| Session | Commands | Expected time | Output in `$HOST_WORK` |
| --- | --- | --- | --- |
| 1. Environment and Nemotron-H | `container.sh env_check.sh`; `container.sh nemotron_h_repro.sh all` | about 2 h | `env/`, `launch_*.log`, `kl.log` |
| 2. Hybrid cache correctness | `container.sh hybrid_cache_tests.sh all` | about 2 h | `hybrid/` |
| 3. Per-path cap, paired | `container.sh paired_path_cap.sh` | 4-6 h | `paired/pool-*/` |
| 4. Prefill and copy bandwidth (optional) | `container.sh prefill_and_copy.sh` | about 3 h | `prefill/`, `copy_bandwidth.json` |

Run session 2's `gsm8k` stage before session 3: a candidate that fails the accuracy check
is not timed. Session 3 runs three pool sizes by default (`POOLS="64 256 default"`); each
is five pairs of server restarts and the same deterministic load. Every script reads its
settings from environment variables listed at the top of the file.

## Running

On the GPU host:

```bash
git clone https://github.com/outerarc-ai/state-bench && cd state-bench
envs/mi300x/container.sh env_check.sh
nohup envs/mi300x/container.sh paired_path_cap.sh > ~/sb-mi300x/paired.log 2>&1 &
tail -f ~/sb-mi300x/paired.log          # survives SSH drops; "=== done" marks the end
```

Collect the outputs without the model cache before releasing the GPU:

```bash
tar -czf mi300x-$(date +%F).tar.gz --exclude=./hf --exclude=./cache -C ~/sb-mi300x .
```

## Reporting

Results go to `results/<date>-mi300x-<name>/` with the environment record, the exact
command and configuration, the raw records and server logs, and a README that labels every
number as measured, spec, assumption, or derived. A paired result reports all four
denominators and states the number of complete pairs.

## Not covered here

SGLang registers some hybrid-cache tests for four GPUs (`test_int8_mamba_checkpoint_e2e.py`,
`test_qwen35_hicache.py`, and the original `test_unified_radix_cache_kl_mamba.py`), and
decode context parallelism needs more than one GPU. Those need a multi-GPU machine and are
not part of these sessions.

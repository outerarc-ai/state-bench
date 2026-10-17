# Paired-run rehearsal on gfx1151

**Result: `state-bench paired` ran end to end against real SGLang servers: ten runs in abba
order, all complete, with the four denominators and paired statistics. Each arm's
server-reported cache hits were identical across its five runs. With a 16-slot state pool
and eight interleaved sessions, `--mamba-max-states-per-path 2` changed the cache hit of 7
of 48 requests (4 more, 3 fewer) and raised the cached share of prompt tokens from 37.3% to
38.7% (measured).** This checks the tool and the run procedure on a shared development GPU.
The latency differences below are a few milliseconds and are not a performance result.
Configuration: [config.json](config.json); report: [report.json](report.json); records and
manifest: [raw/](raw); script: [run.sh](run.sh); per-request comparison:
[analyze.py](analyze.py).

## Setup

- Device: Strix Halo (Radeon 8060S, gfx1151), shared host; no other GPU process at the
  start of the round. Image `rocm/sgl-dev:v0.5.21-rocm724-gfx1151-20261008`, SGLang
  `9f5d871`.
- Model: Qwen3.5-4B (Gated DeltaNet hybrid). State pool 16 slots
  (`--max-mamba-cache-size 16`), `--max-running-requests 2`, `extra_buffer` with track
  interval 128, Triton backends, chunked prefill 2048, context 16,384.
- Arms: baseline as above; candidate adds `--mamba-max-states-per-path 2`. Five pairs,
  abba order, server restarted for every run, first 2 requests of each run dropped.
- Load (assumption): 8 sessions share a 1,024-token prefix; each of 6 turns appends
  256–1,024 random tokens; one request at a time, round-robin over sessions, 16 output
  tokens. The generator is `load.py` inside [run.sh](run.sh) (seed 17), written before
  `state-bench paired load` existed; the two draw tokens in a different order, so their
  prompts differ.
- SLO for the on-time count: TTFT ≤ 10 s (assumption; every request met it).

## Results (measured)

Every run: offered 46, admitted 46, completed 46, on time 46; on time and valid not
checked (the load does not check output).

| Metric | Baseline (median of 5) | Candidate (median of 5) | Median pair difference | 95% CI | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| TTFT p50 (s) | 1.0789 | 1.0721 | −0.0031 | [−0.0170, 0.0035] | no significant difference |
| TTFT p90 (s) | 2.1948 | 2.1960 | −0.0066 | [−0.0241, 0.0114] | no significant difference |
| E2E p50 (s) | 2.0715 | 2.0659 | −0.0040 | [−0.0208, −0.0017] | candidate lower |
| E2E p90 (s) | 3.0267 | 3.0198 | −0.0069 | [−0.0276, 0.0062] | no significant difference |
| Cached token share | 0.3731 | 0.3874 | +0.0144 | [0.0144, 0.0144] | candidate higher |
| Per-request TTFT difference (s) | — | — | −0.0014 | [−0.0149, 0.0055] | no significant difference |

The cached share is deterministic per arm, so its interval has zero width. Requests whose
cache hit differs (prompt tokens; cached tokens baseline → candidate):

| Request | Prompt | Baseline | Candidate |
| --- | ---: | ---: | ---: |
| s4-t1 | 2,338 | 1,024 | 1,728 |
| s0-t2 | 3,263 | 2,368 | 1,024 |
| s1-t2 | 3,405 | 3,008 | 1,024 |
| s2-t2 | 2,402 | 1,984 | 1,024 |
| s4-t2 | 3,238 | 1,024 | 2,304 |
| s0-t3 | 4,128 | 1,024 | 3,200 |
| s1-t3 | 3,935 | 1,024 | 3,392 |

In the last two turns (t4, t5), every request in both arms hits only the shared
1,024-token prefix: with 16 slots for 8 sessions, little of each session's own history
survives. The direction (a small cap helps a little when the pool is this
tight) matches the simulation in
[../2026-10-16-sim-path-cap](../2026-10-16-sim-path-cap/summary.md); the size of the effect
on this workload is not comparable with it.

## What the rehearsal changed in the tooling

- **Server logs.** The round archived records and the manifest but not the per-run server
  logs, so the cap's effect is evidenced only by the cached-token difference.
  `envs/mi300x/paired_path_cap.sh` keeps the server logs in the run directory and checks
  the server arguments in them.
- **Device check.** This rehearsal recorded `rocm-smi --showpids` output without an idle
  pattern. `envs/mi300x/env_check.sh --idle` prints `device idle` for use with
  `device_idle_pattern`.
- **Shutdown.** Every server exited with −9 about 6 s after the stop: SGLang ends its own
  process tree with SIGKILL on SIGTERM. Expected; no change needed.
- **Container restarts.** The container had stopped between rounds (exit 137, two days
  earlier; cause unknown). The round script now checks and restarts its own container and
  stops if another job holds the GPU.

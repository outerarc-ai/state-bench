# Measurement protocol

This protocol governs every performance number published from this repository, in
documents, issues, or pull requests elsewhere. `state-bench paired` runs and reports
alternating repetitions as described below (see [Tooling](#tooling)); the rules also
apply to manual runs.

## Label every number

Each reported number carries one of four labels:

| Label | Meaning |
| --- | --- |
| measured | Produced by a run under this protocol |
| spec | A vendor's peak or nominal figure |
| assumption | A chosen input, such as a synthetic-trace parameter |
| simulated | Output of a model or simulator |

Simulated and measured numbers are reported in separate tables.

## Run conditions

1. **Exclusive device.** Check before and after each run that no other process uses the
   device; record clock and power settings.
2. **Alternating repetitions.** Run baseline and candidate as A, B, A, B, … for at least
   five pairs. Restart the server and clear caches between runs; exclude warm-up requests.
3. **Correctness first.** A candidate that changes cache behavior must pass the engine's
   KL-consistency tests and a small accuracy check before it is measured.

## Statistics

- Treat the per-pair difference (candidate minus baseline) as the sample. Report the
  median and a 95% bootstrap confidence interval.
- Compare request-level metrics such as TTFT request by request on the same trace.
- When the interval contains zero, report "no significant difference".

## Denominators

Report all four for every run, and state the denominator of every ratio:

| Denominator | Meaning |
| --- | --- |
| offered | Requests sent |
| admitted | Requests accepted by the server |
| completed | Requests that finished |
| on-time valid | Requests that finished within the SLO with correct output |

## Bound before measuring

Before optimizing, estimate the largest request-level gain with Amdahl's law. A measured
gain above that bound indicates a measurement problem. Operating points where the
candidate is slower, such as no reuse or ample capacity, are reported alongside the gains.

## Result layout

Each run produces a directory `results/<YYYY-MM-DD>-<name>/` containing:

- `config.yaml`: image tag, engine commit, launch flags, device and driver information.
- `raw/`: per-request records.
- `summary.md`: findings, with every number labeled.

Functional checks (correctness runs on development hardware) use the same layout and
state in `summary.md` that they are not performance results. Result directories are
append-only. A run that supersedes an earlier conclusion gets a
new directory whose summary links to the old one.

## Tooling

`state-bench paired` implements the run conditions, statistics, and denominators above
for any engine that can be started and stopped from the shell.

```bash
state-bench paired plan --pairs 5 --order abba        # print the run order
state-bench paired run run.json -o results/<date>-<name>
state-bench paired report results/<date>-<name> > results/<date>-<name>/raw/report.json
```

A run configuration names the server start (and optional stop) command for each arm, a
readiness command, a load command, an optional device check, the SLO, and the number of
warm-up requests to drop. Its full format is in the docstring of
[`bench/paired.py`](../src/state_bench/bench/paired.py). For every run, `run`:

1. records the device check's output;
2. starts the arm's server in its own process group and waits for readiness;
3. runs the load command, which writes one JSON record per request
   ([`bench/records.py`](../src/state_bench/bench/records.py));
4. stops the server and records the device check again.

`report` drops the warm-up requests, counts the four denominators per run, and compares
TTFT and end-to-end percentiles and the cached-token share over complete pairs: the
median per-pair difference with a seeded 95% bootstrap interval. It also gives the median
of per-request TTFT differences on requests that completed in both runs of a pair. It
warns about failed runs, a device check without the configured idle pattern, and fewer
than five complete pairs. `on_time_valid` is reported only when the load command checked
output correctness; otherwise it is `null` and `on_time` gives the count without it.

For SGLang, `state-bench paired load URL -o {records}` is a ready-made load command: a
deterministic multi-turn workload ([`bench/sglang_load.py`](../src/state_bench/bench/sglang_load.py))
in which sessions share a prefix and grow as append chains, sent one request at a time in
round-robin order over the native `/generate` endpoint with streaming. It records TTFT,
end-to-end time, and the server-reported cached tokens. Its shape (sessions, turns,
prefix and turn lengths, seed) is an assumption and is stored in the run configuration.
It does not check output correctness, so `on_time_valid` stays `null`; the
correctness gate of the run conditions is a separate step. A replay driver that sends a
recorded trace's requests on its recorded schedule is planned.

[`envs/mi300x`](../envs/mi300x/README.md) applies this protocol on one MI300X with SGLang's
ROCm image.


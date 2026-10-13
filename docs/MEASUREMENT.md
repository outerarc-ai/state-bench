# Measurement protocol

This protocol governs every performance number published from this repository, in
documents, issues, or pull requests elsewhere. Tooling that enforces it is in
development; until then the rules apply to manual runs.

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

Result directories are append-only. A run that supersedes an earlier conclusion gets a
new directory whose summary links to the old one.

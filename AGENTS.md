# Repository guidance

The implementation, tests, and documentation in this repository define state-bench's
behavior. Keep changes scoped to this project.

## Implementation and verification

- Python 3.11+, src layout, `uv.lock`. Install with `uv sync --locked --group dev`.
- Trace generation is deterministic. Do not introduce wall-clock time, unseeded
  randomness, or unordered iteration into anything that is written to a trace.
- Traces never carry prompt or output text. Prefix sharing is expressed only through
  block hashes and segment boundaries.
- Do not change the trace format without bumping `SCHEMA_VERSION` and updating
  `docs/TRACE_FORMAT.md`.
- Label every number as measured, spec, assumption, or simulated. Do not present
  simulated or synthetic results as measurements.
- Engine behavior notes cite the commit read and `file:line` references; re-read the
  source rather than relying on earlier notes when the commit changes.
- When citing an upstream issue or pull request, check its current state first and state
  it (open, draft, merged, closed).
- Validate with `uv run ruff check .`, `uv run ruff format --check .`, and
  `uv run pytest --cov=state_bench --cov-fail-under=90`.

See docs/README.md for the documentation index.

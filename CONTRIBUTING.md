# Contributing to state-bench

Thanks for helping improve state-bench. Please open an issue before changing the trace
format, a public command, or the measurement protocol, so compatibility and the effect on
published results can be discussed first.

## Development setup

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/outerarc-ai/state-bench.git
cd state-bench
uv sync --locked --group dev
```

Run the project checks before opening a pull request:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=state_bench --cov-fail-under=90
```

CI runs the same checks on Python 3.11, 3.12, and 3.13, plus a synthetic trace round trip
through the CLI and a package build.

## Pull requests

- Keep generated traces deterministic: the same inputs must produce the same bytes.
- Changes to the trace format bump `SCHEMA_VERSION` and update [Trace format](docs/TRACE_FORMAT.md).
- New policies or engine notes cite the engine commit they were read from, with
  `file:line` references.
- Label every number in documentation and pull request descriptions as measured, spec,
  assumption, or simulated, following the [measurement protocol](docs/MEASUREMENT.md).
- Include tests for behavior changes. Prefer small cases whose expected result can be
  computed by hand.

## Data

Contribute only traces you generated or recorded yourself, or public datasets whose
license permits redistribution. Recorded traces must not contain prompt or output text.
Each importer documents its source and license.

state-bench is released under the [Apache License 2.0](LICENSE); contributions are
accepted under the same license.

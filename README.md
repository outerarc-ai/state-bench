# state-bench: trace-driven evaluation of state caching for agentic LLM serving

[![CI](https://github.com/outerarc-ai/state-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/outerarc-ai/state-bench/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

state-bench studies how LLM inference engines reuse cached state across requests in agentic workloads. Hybrid models such as Qwen3.5 and Nemotron-H keep a recurrent (Mamba/GDN) state alongside the attention KV cache, and a prefix can be resumed only where a state snapshot exists. Where engines place, keep, and evict those snapshots decides how much of a long agent session is recomputed.

The project provides a content-free trace format for agent sessions, a deterministic synthetic trace generator, documentation of current engine behavior, and a simulator that compares snapshot placement policies on a trace. A paired replay harness for OpenAI-compatible servers is in development. Measurements target AMD Instinct GPUs with ROCm; the trace tooling runs on any CPU.

## What it provides

- **Content-free traces.** Requests are recorded as chained 64-token block hashes, message boundaries, and timing. Prefix reuse is preserved without storing prompts, so recorded sessions can be shared.
- **Reproducible synthetic sessions.** Append, fork, retry, and rewrite patterns are generated from a seed. The same parameters produce byte-identical files, and every parameter is labeled as an assumption.
- **Machine-checkable artifacts.** Traces are validated against a versioned JSON Schema and cross-record rules. Every command is deterministic, returns a meaningful exit code, and emits JSON where output is meant to be consumed by other tools.
- **Policy simulation.** `state-bench sim` replays a trace under a snapshot placement policy and state-pool size and reports recomputed tokens. Policies from SGLang, ATOM, and vLLM are implemented from their documented rules with hand-checked tests, alongside an offline reference that bounds what any placement can save.
- **Documented engine behavior.** [SGLang Mamba checkpoints](docs/SGLANG_MAMBA_CHECKPOINTS.md) describes where SGLang currently places, inserts, and evicts recurrent-state checkpoints, with source references.

Synthetic traces exercise policies; they do not measure them. Performance results will be reported only from controlled runs on real hardware, following the [measurement protocol](docs/MEASUREMENT.md).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/outerarc-ai/state-bench.git
cd state-bench
uv sync --locked --group dev
uv run state-bench --help
```

## Quick start

```sh
uv run state-bench synth --preset mixed --sessions 8 --seed 0 -o mixed.jsonl
uv run state-bench validate mixed.jsonl
uv run state-bench stats mixed.jsonl
uv run state-bench sim mixed.jsonl --state-slots none --state-slots 64
uv run state-bench schema > trace.schema.json
```

`stats` reports the share of input tokens that an ideal cache keeping every parent request could reuse. It is an upper bound derived from the trace, not a measurement of any engine. `sim` output is labeled `simulated` for the same reason.

## Status

| Component | Status |
| --- | --- |
| Trace format v0.1 and validation | Available |
| Synthetic generator (append, fork, retry, rewrite) | Available |
| SGLang checkpoint behavior notes | Available, read at SGLang `166b4ae7f` |
| Importers for public agent traces | Licenses and fields verified ([Public agent traces](docs/PUBLIC_TRACES.md)); importer in development |
| Recording proxy for local agent sessions | Available (v0: OpenAI-compatible chat proxy, `state-bench capture`; lineage derived from prefixes) |
| Snapshot placement simulator | Available (v0: sequential, unlimited KV) |
| Policies: SGLang current and DCP mask, prompt-end anchor, Marconi-style admission, recent-checkpoint retention, offline reference | Available |
| Value-based admission and eviction at message boundaries | Available (v0, simulation only) |
| Paired replay and statistics against live servers | Planned |

## Documentation

| Guide | Contents |
| --- | --- |
| [Overview](docs/README.md) | Documentation index |
| [Trace format](docs/TRACE_FORMAT.md) | Records, fields, hashing, and validation rules |
| [Public agent traces](docs/PUBLIC_TRACES.md) | Licenses, fields, and limits of public trace corpora |
| [Synthetic traces](docs/SYNTHETIC_TRACES.md) | Patterns, parameters, and limits of synthetic data |
| [Measurement protocol](docs/MEASUREMENT.md) | How results are produced, labeled, and reported |
| [Snapshot placement simulator](docs/SIMULATOR.md) | Model, options, output, and limits |
| [SGLang Mamba checkpoints](docs/SGLANG_MAMBA_CHECKPOINTS.md) | Current placement, insertion, and eviction rules |
| [Related work](docs/RELATED_WORK.md) | Upstream discussions and implementations this project builds on |

Issues and pull requests are welcome. See [Contributing](CONTRIBUTING.md) for setup and review expectations. state-bench is released under the [Apache License 2.0](LICENSE).

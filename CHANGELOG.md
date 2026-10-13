# Changelog

## Unreleased

- Snapshot placement simulator (`state-bench sim`, `state_bench.sim`) with an LRU state
  pool and the `sglang_current` policy, checked against hand-derived cases.
- Trace format 0.1: JSONL records with chained 64-token block hashes, message boundaries,
  and timing; JSON Schema and cross-record validation.
- Synthetic generator with append, fork, retry, and rewrite patterns and presets.
- `state-bench` CLI with `synth`, `validate`, `stats`, and `schema` commands.
- Documentation of SGLang Mamba checkpoint placement, insertion, and eviction at commit
  `166b4ae7f`, a measurement protocol, and a related-work survey.

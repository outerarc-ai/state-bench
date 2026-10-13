# Changelog

## Unreleased

- Snapshot placement simulator (`state-bench sim`, `state_bench.sim`) with a state pool,
  pluggable placement and eviction, and lineage-aware prefix matching. Policies:
  `sglang_current`, `sglang_absolute_mask`, `prompt_end_anchor`, `marconi_admission`,
  `pin_recent`, and the `offline_oracle` reference, checked against hand-derived cases.
- Simulated reproduction of the donation-mask regression fixed by SGLang #40907
  (`results/2026-10-13-sim-donation-mask`).
- Trace format 0.1: JSONL records with chained 64-token block hashes, message boundaries,
  and timing; JSON Schema and cross-record validation.
- Synthetic generator with append, fork, retry, and rewrite patterns and presets.
- `state-bench` CLI with `synth`, `validate`, `stats`, and `schema` commands.
- Documentation of SGLang Mamba checkpoint placement, insertion, and eviction at commit
  `166b4ae7f`, a measurement protocol, and a related-work survey.

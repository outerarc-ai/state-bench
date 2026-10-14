# Changelog

## Unreleased

- Replay materialization (`state-bench materialize`, `state_bench.replay`): turn a
  content-free trace into a token-id stream that reproduces its prefix reuse (shared
  blocks and `append`/`retry` lineage), emitted as Autobench timestamped JSONL with a
  `.meta.json` sidecar and a prefix-fidelity check. Token identity is an assumption;
  prompt structure and timing inherit the source trace.
- Snapshot placement simulator (`state-bench sim`, `state_bench.sim`) with a state pool,
  pluggable placement and eviction, and lineage-aware prefix matching. Policies:
  `sglang_current`, `sglang_absolute_mask`, `prompt_end_anchor`, `marconi_admission`,
  `pin_recent`, `boundary_value`, and the `offline_oracle` reference, checked against
  hand-derived cases.
- Simulated reproduction of the donation-mask regression fixed by SGLang #40907
  (`results/2026-10-13-sim-donation-mask`).
- Simulated policy comparison on held-out synthetic traces
  (`results/2026-10-13-sim-policy-comparison`).
- License and field review of the kv-cache-tester and AgentX v1.0 trace corpora
  (`docs/PUBLIC_TRACES.md`).
- Trace format 0.1: JSONL records with chained 64-token block hashes, message boundaries,
  and timing; JSON Schema and cross-record validation.
- Synthetic generator with append, fork, retry, and rewrite patterns and presets.
- `state-bench` CLI with `synth`, `validate`, `stats`, and `schema` commands.
- Documentation of SGLang Mamba checkpoint placement, insertion, and eviction at commit
  `166b4ae7f`, a measurement protocol, and a related-work survey.

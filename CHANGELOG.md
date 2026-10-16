# Changelog

## Unreleased

- `state-bench import weka` (`state_bench.traces.weka`): converts kv-cache-tester and
  AgentX traces to the state-bench format with hash scheme `weka-local-v1`, deriving
  parent links, patterns, and tool waits; streams one trace at a time.
- Idle-gap distribution of the 739 kv-cache-tester traces
  (`results/2026-10-15-public-trace-idle-gaps`).
- Simulated policy comparison on 120 recorded kv-cache-tester sessions
  (`results/2026-10-15-sim-policy-public-traces`).
- `state-bench paired` (`state_bench.bench`): alternating baseline/candidate runs driven by
  shell commands, per-request records, the four denominators, and paired bootstrap
  statistics, following `docs/MEASUREMENT.md`.
- `envs/mi300x`: scripts for one MI300X in SGLang's ROCm image: environment record and
  idle check, the Nemotron-H on ROCm reproduction, single-GPU hybrid-cache correctness
  checks, paired per-path cap runs at several state-pool sizes, prefill curve, and
  host-device copy bandwidth.
- `state-bench paired load` (`state_bench.bench.sglang_load`): a deterministic
  multi-turn load for SGLang's `/generate` endpoint that writes per-request records.
- `state-bench sim --path-cap-min-usage` applies the per-path cap only while the state
  pool is nearly full, and a sweep of it
  (`results/2026-10-16-sim-pressure-gated-cap`).
- `state-bench sim --max-states-per-path` (`SimConfig.max_states_per_path`) models
  SGLang's per-path snapshot cap, and a sweep of it on recorded traces
  (`results/2026-10-16-sim-path-cap`).
- `boundary_value` accepts `eviction=lru` to keep its placement and evict by recency,
  and an ablation separating eviction from placement on recorded traces
  (`results/2026-10-16-sim-retention-ablation`).
- Simulated pool-size sweep in slots per live session, with state slots per session
  under SGLang's default memory split for six hybrid models
  (`results/2026-10-15-sim-state-pool-pressure`).
- Restore-versus-recompute estimate for a hybrid model on MI300X, from spec values and
  stated assumptions (`results/2026-10-15-restore-vs-recompute-estimate`).
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

# Snapshot placement simulator

`state-bench sim` replays a trace and reports, for a snapshot placement policy and a
state-pool size, how many prompt tokens each request must recompute. Its output is
labeled `provenance: simulated`; it compares policies under stated assumptions and is not
a measurement of any engine.

## Usage

```sh
uv run state-bench synth --preset mixed --sessions 64 --seed 1 -o mixed.jsonl
uv run state-bench sim mixed.jsonl --state-slots none --state-slots 256 --state-slots 64
```

`--state-slots` and `--policy` can be repeated; the command prints one JSON row per
combination. Engine parameters:

| Option | Default | Meaning |
| --- | --- | --- |
| `--cache-chunk-size` | 64 | Granularity at which a state can be taken: max(model chunk size, page size) |
| `--state-chunk-size` | 64 | The model's own recurrent chunk size |
| `--page-size` | 1 | Radix cache page size |
| `--chunked-prefill-size` | 2048 | Prefill chunk length |
| `--track-interval` | 256 | Decode snapshot interval |
| `--no-output-roundtrip` | off | Assume replies are not replayed token for token in the next request |
| `--per-request` | off | Include one record per request |

From the API:

```python
from state_bench.sim import SimConfig, get_policy, simulate
from state_bench.traces import load_trace

result = simulate(
    load_trace("mixed.jsonl"), get_policy("sglang_current"), SimConfig(state_slots=64)
)
print(result.summary())
```

## Output

| Field | Meaning |
| --- | --- |
| `prompt_tokens` | Sum of prompt lengths |
| `recompute_tokens` | Prompt tokens computed because no usable state covered them |
| `recompute_ratio` | `recompute_tokens / prompt_tokens` |
| `retreat_tokens` | Tokens with cached KV that were recomputed because no state lay at the end of the KV hit |
| `snapshots_written` | New snapshots stored |
| `evictions` | Snapshots evicted from the state pool |
| `state_peak` | Largest number of snapshots held at once |

Per request, `kv_hit` is the cached KV prefix, `usable_prefix` the deepest usable state
within it, and `recompute_tokens = prompt_tokens - usable_prefix`.

## Model

The simulator keeps two caches:

- **KV coverage.** Version 0 assumes unlimited KV capacity. KV is added only up to each
  inserted snapshot, as in SGLang, where inserted KV stops at the latest snapshot.
- **State pool.** Snapshots at token positions, with a fixed number of slots and
  least-recently-used eviction. Using a snapshot refreshes only that snapshot.

For each request, in arrival order:

1. Find the KV hit, capped at `prompt_tokens - 1` because at least one token is computed.
2. Take the deepest snapshot within the KV hit as the usable prefix.
3. Ask the policy for prefill snapshot positions, starting from the usable prefix, and
   insert them.
4. Ask the policy for a completion snapshot. A position beyond the prompt lies in the
   reply; it is stored only for the request's first `append` child and only when replies
   round-trip.

## Policies

| Name | Description |
| --- | --- |
| `sglang_current` | SGLang `extra_buffer` placement as documented in [SGLang Mamba checkpoints](SGLANG_MAMBA_CHECKPOINTS.md) |

A policy implements `plan_prefill(req, start, kv_hit, cfg)` and `plan_finish(req, cfg)`.
Each case in `tests/test_sim_sglang.py` derives its expected positions by hand from the
documented rules.

## Limits

- **Block resolution.** Traces carry 64-token block hashes, so equality inside a block is
  invisible. A snapshot is matched only through the block that contains it, and KV that
  ends inside a block is matched only by requests sharing that whole block. The simulator
  can therefore miss reuse that an engine would find, such as a snapshot in the partial
  last block of an earlier prompt; it never reports reuse that is not there.
- **No concurrency.** Requests run one at a time. In-flight state slots, which SGLang
  holds per running request, are not counted against the pool.
- **No KV capacity.** KV eviction and its interaction with the state pool are not modeled.
- **No timing.** The output counts tokens. Converting recomputed tokens into latency needs
  a cost model, and request-level gains need measurement under the
  [measurement protocol](MEASUREMENT.md).

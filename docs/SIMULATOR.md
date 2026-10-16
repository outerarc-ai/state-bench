# Snapshot placement simulator

`state-bench sim` replays a trace and reports, for a snapshot placement policy and a
state-pool size, how many prompt tokens each request must recompute. Its output is
labeled `provenance: simulated`; it compares policies under stated assumptions and is not
a measurement of any engine.

## Usage

```sh
uv run state-bench synth --preset mixed --sessions 64 --seed 1 -o mixed.jsonl
uv run state-bench sim mixed.jsonl --state-slots none --state-slots 256 --state-slots 64
uv run state-bench sim mixed.jsonl --policy sglang_current --policy pin_recent:k=2 \
    --policy offline_oracle --state-slots 64
```

`--state-slots` and `--policy` can be repeated; the command prints one JSON row per
combination. A policy takes parameters after a colon, for example
`pin_recent:k=2,active_s=none`. Engine parameters:

| Option | Default | Meaning |
| --- | --- | --- |
| `--cache-chunk-size` | 64 | Granularity at which a state can be taken: max(model chunk size, page size) |
| `--state-chunk-size` | 64 | The model's own recurrent chunk size |
| `--page-size` | 1 | Radix cache page size |
| `--chunked-prefill-size` | 2048 | Prefill chunk length |
| `--track-interval` | 256 | Decode snapshot interval |
| `--no-output-roundtrip` | off | Assume replies are not replayed token for token in the next request |
| `--first-chunk-cut` | `none` | `hashed` cuts the first of several prefill chunks at a per-request length, standing in for a shared prefill budget running out (assumption) |
| `--max-states-per-path` | none | Per-path snapshot cap, as SGLang's `--mamba-max-states-per-path`; applies to every policy (see Model) |
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
| `snapshots_rejected` | Snapshots the policy declined to admit into a full pool |
| `evictions` | Snapshots evicted from the state pool |
| `state_peak` | Largest number of snapshots held at once |

Per request, `kv_hit` is the cached KV prefix, `usable_prefix` the deepest usable state
within it, and `recompute_tokens = prompt_tokens - usable_prefix`.

## Model

The simulator keeps two caches:

- **KV coverage.** Version 0 assumes unlimited KV capacity. KV is added only up to each
  inserted snapshot, as in SGLang, where inserted KV stops at the latest snapshot.
- **State pool.** Snapshots at token positions, with a fixed number of slots. When the
  pool is full, the policy's priority picks the victim; a new snapshot whose priority is
  lower than every resident's is not admitted. The default priority is least recently
  used, and using a snapshot refreshes only that snapshot.
- **Per-path cap (optional).** With `--max-states-per-path N`, each newly written
  snapshot triggers the eviction of the shallowest snapshots on its path beyond `N`, as
  SGLang does (`mem_cache/unified_cache/components/mamba.py:251-306` at `c4870d98`). The
  new snapshot, fork points, and the snapshot the current request resumed from are
  skipped. A fork point is recognized only at a block boundary, from the distinct blocks
  cached after it; SGLang's tree also sees forks inside a block, so the simulator may
  evict a few snapshots that SGLang would keep.

For each request, in arrival order:

1. Find the KV hit, capped at `prompt_tokens - 1` because at least one token is computed.
2. Take the deepest snapshot within the KV hit as the usable prefix.
3. Ask the policy for prefill snapshot positions, starting from the usable prefix, and
   insert them.
4. Ask the policy for completion snapshots. A position beyond the prompt lies in the
   reply; it is stored only for the request's first `append` child and only when replies
   round-trip.

Prefixes are matched through shared full blocks and through `append` and `retry`
lineage, which guarantees that a request starts with its ancestor's whole prompt. Prefix
sharing that neither shows, such as a fork that diverges inside a block, is not
credited.

## Policies

| Name | Placement | Eviction | Source |
| --- | --- | --- | --- |
| `sglang_current` | SGLang `extra_buffer` rules: one snapshot per prefill chunk, aligned to the chunk start; branching point; decode snapshots every `track_interval` | LRU | [SGLang Mamba checkpoints](SGLANG_MAMBA_CHECKPOINTS.md) |
| `sglang_absolute_mask` | As `sglang_current`, with the absolute-grid donation mask SGLang uses under DCP | LRU | SGLang [#39115](https://github.com/sgl-project/sglang/pull/39115), [#40907](https://github.com/sgl-project/sglang/pull/40907) |
| `prompt_end_anchor` | One snapshot at each prompt end, rounded down to the grid; nothing else | LRU | Prompt-end anchor of ATOM [#2045](https://github.com/ROCm/ATOM/pull/2045), interval and demand checkpoints off |
| `marconi_admission` | Last state of each request (prompt and reply), plus a snapshot at the last grid point of a KV hit that runs past the usable state | LRU | vLLM [#37898](https://github.com/vllm-project/vllm/pull/37898), after [Marconi](https://arxiv.org/abs/2411.19379) |
| `pin_recent` | As `sglang_current` | LRU, except that the `k` newest snapshots of each active conversation chain are evicted last; parameters `k` (default 4) and `active_s` (default 120 s, assumption) | Retention proposed in SGLang RFC [#40865](https://github.com/sgl-project/sglang/issues/40865) |
| `boundary_value` | End of every message in the prompt and of the reply, rounded down to the grid; optional KV-hit demand point | By estimated value: reuse probability from the session's history so far × tokens saved × decay over a horizon scaled to the session's request gap; low-value snapshots are not admitted. `eviction=lru` keeps the placement and evicts by recency instead; `reply_end=false` drops the reply-end snapshot | This project; see below |
| `offline_oracle` | Exactly the positions later requests can use, known in advance | Furthest next use first (Belady), with bypass | Reference only; not implementable online |

`offline_oracle` bounds every other policy when the pool is unlimited: no policy in this
simulator can recompute fewer tokens. With a limited pool it is a strong reference, not a
proven optimum, because snapshots differ in the tokens they save.

`boundary_value` makes two assumptions to check before reading its results. It takes
several snapshots per prefill chunk, which needs an engine that can capture interior
states of a forward pass (ATOM #2045 does; SGLang records one per chunk). And in the
synthetic generator every branch point, whether a fork, a rewrite, or a shared system
prompt, falls on a message boundary, which favors boundary placement; recorded traces
are needed to test that premise. Parameters (`gap_factor`, `tau_s`) are assumptions;
the module docstring gives the value formula.

A policy subclasses `state_bench.sim.Policy` and overrides `plan_prefill(req, start,
kv_hit, cfg)`, `plan_finish(req, cfg)`, `priority(snapshot, sim)`, or `bind(sim)`. The
cases in `tests/test_sim_sglang.py` and `tests/test_sim_policies.py` derive their expected
positions by hand.

## Limits

- **Block resolution.** Traces carry 64-token block hashes, so equality inside a block is
  visible only through lineage. The simulator can miss reuse that an engine would find,
  such as a fork that diverges inside a block; it never reports reuse that is not there.
- **No concurrency.** Requests run one at a time. In-flight state slots, which SGLang
  holds per running request, are not counted against the pool, and chunk boundaries do
  not depend on other requests unless `--first-chunk-cut hashed` is set.
- **No KV capacity.** KV eviction and its interaction with the state pool are not modeled.
- **No timing.** The output counts tokens. Converting recomputed tokens into latency needs
  a cost model, and request-level gains need measurement under the
  [measurement protocol](MEASUREMENT.md).

# Replay materialization

A trace records chained 64-token block hashes, never tokens (see the
[trace format](TRACE_FORMAT.md)). A serving engine, though, takes token ids, so replaying
a recorded session on real hardware needs a token stream that reproduces the trace's
prefix reuse. `state-bench materialize` builds one.

The command reads a trace and writes a token-id stream in the SGLang Simulator
"Autobench" timestamped JSONL format, so the same trace can drive both the
[simulator](SIMULATOR.md) and a live server. Token *identity* is an arbitrary placeholder
(`provenance: assumption`); prefix sharing, prompt lengths, and arrival timing are copied
from the trace and inherit its provenance.

## Usage

```sh
uv run state-bench synth --preset mixed --sessions 8 --seed 0 -o mixed.jsonl
uv run state-bench materialize mixed.jsonl -o replay.jsonl --vocab-size 151936
```

`--vocab-size` is the target tokenizer's vocabulary size; every emitted id lies in
`[0, vocab-size)` so the engine's embedding lookup accepts it. The command runs a
prefix-fidelity check by default and exits non-zero if it fails; `--no-check` skips it.

From the API:

```python
from state_bench.replay import check_prefix_fidelity, materialize, write_replay
from state_bench.traces import load_trace

trace = load_trace("mixed.jsonl", check=False)
mat = materialize(trace, vocab_size=151936)
assert check_prefix_fidelity(trace, mat) == []
write_replay(trace, "replay.jsonl", vocab_size=151936)  # JSONL plus a .meta.json sidecar
```

## What is reproduced

Block hashes are chained, so two requests that share leading blocks share those hashes.
Materialization assigns each block-hash prefix one deterministic token sequence, memoized
by that prefix: equal hashes give equal tokens, so an engine's prefix cache sees the reuse
the trace records, and (up to hash collisions) no reuse it does not.

Two relations are reproduced:

- **Shared blocks.** Requests with the same leading block hashes get the same leading
  tokens, across sessions as well as within one (for example a shared system prompt).
- **`append`/`retry` lineage.** The trace guarantees such a request starts with its
  parent's whole prompt, tail included. The child's tokens are copied from the parent's
  materialized prompt and continued from the parent's end, so a parent that ends mid-block
  is still shared exactly -- matching how the simulator credits it
  ([`sim/core.py`](../src/state_bench/sim/core.py), which widens a lineage child's KV hit
  to the ancestor's whole prompt).

A `fork` or `rewrite` is not a whole-prompt link and is reproduced only at block
resolution.

## Output

The Autobench JSONL has one row per request, in global arrival order:

| Field | Meaning |
| --- | --- |
| `prompt` | Token ids, length equal to the trace's `prompt_tokens` |
| `prompt_len` | `len(prompt)` |
| `output_len` | The trace's output length; omitted when the trace recorded none |
| `timestamp` | Arrival in milliseconds from the trace's origin (session start + relative arrival) |

`output_len` is omitted rather than written as `0`, because the SGLang Simulator loader
falls back to its own default when the field is absent; a recorded `0` must not be sent as
an explicit value. The reader is
`benchmark/simulator/bench_serving.py --dataset-name autobench --use-trace-timestamps` in
the [SGLang simulator](https://github.com/sgl-project/sglang) tree, which sorts rows by
`timestamp` and normalizes to the first one.

Alongside the JSONL, `write_replay` writes `<output>.meta.json` recording the token
scheme, the vocabulary size, the provenance labels, and the source trace's generator and
size.

## Fidelity check

`check_prefix_fidelity` verifies the materialized tokens against the trace. Each prompt
must have exactly `prompt_tokens` ids in range. For each pair of requests -- all pairs
when they number few enough, otherwise parent/child pairs -- the shared token prefix must
equal the block-level share, widened by any whole-prompt `append`/`retry` relation. Any
mismatch is reported. On a validated trace the check passes by construction; it exists to
catch a materializer that drifts from the trace it is supposed to reproduce.

## Limits

- **Block resolution.** Inside a 64-token block only lineage is exact. A fork or rewrite
  that diverges mid-block is reproduced at block granularity, as in the
  [simulator](SIMULATOR.md).
- **Hash collisions.** Two different blocks could reduce to the same token sequence at a
  small vocabulary; the fidelity check catches it because it compares against the trace.
- **Placeholder tokens.** The ids are not the real prompt. They reproduce reuse structure,
  not content, so engine output is not meaningful; replay targets prefix-cache and
  state-reuse behavior, not answer quality.
- **No driver.** Materialization produces the token stream; a driver that replays it
  against a live OpenAI-compatible server is not part of this command.

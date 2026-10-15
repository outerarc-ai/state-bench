# Trace format

Version 0.1. The JSON Schema is printed by `state-bench schema` and defined in
[`schema.py`](../src/state_bench/traces/schema.py).

A trace records the structure and timing of requests, never their content. Each
request's input is split into 64-token blocks and hashed as a chain, so two requests
share a block hash only if they share every token up to the end of that block.
Prefix reuse can therefore be analyzed and replayed without the original text.

## Records

A trace is a JSONL file. The first line is a `header`; the remaining lines are
`session` and `request` records.

| Record | Fields |
| --- | --- |
| `header` | `schema_version`, `block_tokens` (always 64), `hash_scheme`, `generator`, `params` |
| `session` | `session_id`, `t_start_s` (relative to the trace), `model_family`, `source`, `license` |
| `request` | `req_id`, `session_id`, `parent_req_id`, `branch_id`, `t_arrival_s` (relative to the session), `prompt_tokens`, `blocks`, `segments`, `output_tokens`, `tool_wait_s`, `pattern` |

### Request fields

- `blocks`: chained block hashes as 16 lowercase hex digits. The last block may be
  partial; its length is `prompt_tokens - 64 * (len(blocks) - 1)`. Only full blocks
  can match across requests.
- `segments`: message boundaries as `{kind, start_tok, end_tok}`, where `kind` is
  `system`, `user`, `assistant`, or `tool_result`. Use `null` when the source does not
  record boundaries. Policies that depend on boundaries should report "not applicable"
  rather than infer them.
- `pattern`: relationship to the parent: `append`, `fork`, `retry`, or `rewrite`.
  Root requests have neither a parent nor a pattern.
- `tool_wait_s`: time from the end of this request's output to the arrival of the next
  request in the chain. Replay tools should use it to drive a closed loop; `t_arrival_s`
  is the nominal schedule under the source's own service times.

### Hash schemes

`hash_scheme` names how block hashes were computed. Hashes from different schemes are not
comparable.

| Scheme | Used by | Definition |
| --- | --- | --- |
| `blake2b64-chain-v1` | `state-bench synth` | BLAKE2b-64 chained over each block's tokens |
| `weka-local-v1` | `state-bench import weka` | BLAKE2b-64 of the trace ID and the source's local block ID; a partial last block hashes the request ID. Equal only within one trace |

Importers map source block IDs one to one and name the source scheme; they never claim
that blocks from different sources match.

## Validation

`state-bench validate` runs two layers:

1. **Structure.** Each line must match the JSON Schema.
2. **Semantics.** Across records:
   - `len(blocks) == ceil(prompt_tokens / 64)`.
   - Segments start at 0, are contiguous, and end at `prompt_tokens`.
   - Sessions are declared; request IDs are unique.
   - A parent exists, belongs to the same session, appears earlier in the file, and does
     not arrive after its child.
   - `pattern` is set exactly when `parent_req_id` is set.
   - `append` extends every full block of its parent; `retry` repeats its parent's blocks.
     `fork` and `rewrite` may share less than one block, so they are not checked at block
     level.
   - Requests within a session appear in non-decreasing arrival order.

Validation establishes that a trace is internally consistent. It does not establish that
a recorded trace is representative of any production workload.

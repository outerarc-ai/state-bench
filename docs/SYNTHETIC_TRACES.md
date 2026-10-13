# Synthetic traces

`state-bench synth` generates agent sessions from a seed. Synthetic traces are useful for
unit-level policy tests and sensitivity sweeps. Every parameter is an assumption and is
recorded in the trace header with `provenance: assumption`; synthetic traces are not a
basis for performance claims.

```sh
uv run state-bench synth --preset mixed --sessions 64 --seed 0 -o mixed.jsonl
```

## Patterns

| Pattern | Next request | Main parameters |
| --- | --- | --- |
| `append` | Parent input + parent output + a new tool result | Turns, output and tool-result lengths, tool wait |
| `fork` | A sub-agent that shares the system prompt or the parent's whole input, plus a task message; the parent resumes after all branches finish | Probability, width, share point, branch turns |
| `retry` | The parent's input sent again unchanged | Probability, retry delay |
| `rewrite` | The first *k* messages kept and the rest replaced by a summary; optionally triggered when the context limit is reached | Probability, *k*, summary length, compaction on overflow |

Presets: `append`, `fork`, `retry`, `rewrite`, and `mixed`. Their probabilities are set in
`synth.preset_params`. For finer control, construct `TraceParams` in Python.

## How content is modeled

Each message is a seed and a length. Its bytes are expanded with `shake_256(seed)` and
hashed in 64-token chained blocks. A message that reappears in a later request yields the
same block hashes, so prefix reuse follows from construction rather than being assigned
afterwards. The same parameters always produce the same bytes.

## Timing

Service time is `nominal_ttft_s + output_tokens / nominal_decode_tps`. The next request
arrives after the service time plus the sampled tool wait. Forked sub-agents run before
the parent's next request, which therefore waits for the slowest branch.

## Current limits

- Branches neither fork nor rewrite.
- A request is retried at most once; the retry carries `tool_wait_s` and the original
  carries `null`.
- Output tokens are assumed to round-trip exactly into the next request. Chat templates
  that drop reasoning or reformat tool calls break this assumption; recorded traces are
  needed to measure that effect.

# Public agent traces

Two public corpora record real agentic coding sessions in a form close to this
repository's [trace format](TRACE_FORMAT.md). This page records their licenses, fields,
and what they can and cannot tell us. Checked on 2026-10-13; follow the links for the
current state.

## Sources

| Corpus | Publisher | Where | License | Size |
| --- | --- | --- | --- | --- |
| kv-cache-tester traces | WEKA ([callanjfox/kv-cache-tester](https://github.com/callanjfox/kv-cache-tester), `traces/`) | GitHub | Apache-2.0 (repository license) | 739 traces, 59,204 requests |
| AgentX v1.0, `cc-traces-weka-062126` | SemiAnalysis ([dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126), [methodology](https://inferencex.semianalysis.com/agentx)) | Hugging Face, not gated | Apache-2.0 (dataset card) | 393 traces, 98,827 requests, one 1.85 GB JSONL file |
| AgentX v1.0, `cc-traces-weka-062126-256k` | SemiAnalysis ([dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)) | Hugging Face, not gated | Apache-2.0 (dataset card) | 393 traces, requests above 256k tokens removed |

Both corpora come from Claude Code sessions captured through an HTTP proxy. Their
publishers state that prompts, code, tool arguments, and tool results were removed and
that only lengths, block hashes, timing, and topology remain. The kv-cache-tester README
states that conversation IDs, timestamps, and agent IDs are anonymized; the AgentX
methodology states that the sessions were opt-in.

This repository does not redistribute either corpus. An importer will read a local copy
and record the source and license in every session record.

## Format

Both use the same JSON layout (one object per trace; AgentX stores one per line).

| Field | Level | Meaning | state-bench field |
| --- | --- | --- | --- |
| `id` | trace | Trace identifier (anonymized) | `session_id` |
| `models` | trace | Models seen in the trace | `model_family` (workload origin, not the served model) |
| `block_size` | trace | Always 64 | `block_tokens` |
| `hash_id_scope` | trace | `local`: IDs are unique within one trace, including its subagents | `hash_scheme` (see below) |
| `tool_tokens`, `system_tokens` | trace, subagent | Length of the tool definitions and system prompt | First boundary of `segments` (only one known) |
| `t` | request | Seconds since the start of the trace | `t_arrival_s` |
| `type` | request | `n` (non-streaming), `s` (streaming), or `subagent` (nested trace with its own `requests`) | Subagent requests join the same session |
| `in`, `out` | request | Input and output tokens | `prompt_tokens`, `output_tokens` |
| `hash_ids` | request | Integer IDs of the prompt's 64-token blocks | `blocks` |
| `input_types`, `output_types`, `stop` | request | Content types of the new input and of the response; stop reason | Not mapped |
| `api_time`, `think_time` | request | Server time and client delay before the request | `think_time` of the next request gives `tool_wait_s` |

Differences between the two builds:

- In kv-cache-tester traces, `in` is a token count and `hash_ids` cover the full blocks of
  the prompt: `len(hash_ids) = floor(in / 64)` in all 5,154 requests of a 60-trace
  sample. In
  the AgentX build, `in` is the number of hash blocks times 64, which its dataset card
  says can exceed the billed prompt for heavily cached contexts.
- AgentX filters sessions (at least 20 requests, recent client versions, at most 10
  concurrent subagents) and drops requests above 990,016 tokens; see the dataset card.

## Observed properties

Measured on the first 60 kv-cache-tester traces (5,000 top-level and 154 subagent
requests). These are properties of
the trace data, not measurements of any engine.

- **Block IDs behave like chained hashes.** For 24,877 request pairs within a trace, no
  pair that differed at some block matched again at a later block. Equal IDs at block
  `k` can therefore be read as an equal prefix of `64 * k` tokens, which is what the
  simulator assumes.
- **Prompts are mostly extended, sometimes cut.** Of 4,938 consecutive top-level request
  pairs on the same model, 420 (8.5%) kept only part of the earlier prompt, as in context
  compaction or a branch.
- **Subagents are nested** inside the parent trace with their own `tool_tokens` and
  `system_tokens` and share the parent's ID namespace.

## What the traces do not record

- **Reply round-trip.** `hash_ids` cover only a request's own prompt, never its reply, so
  the traces cannot show whether a reply reappears token for token in the next request.
  Whether it does depends on the served model's chat template, for example whether it
  removes earlier reasoning. The simulator's `output_roundtrip` remains an assumption.
- **Message boundaries.** Apart from the tool and system prefix and each request's prompt
  end, boundaries inside the new input (reply versus tool result) are not recorded.
  Branch points can be located at block resolution and compared with the known
  boundaries, but not with all of them.
- **Sharing across conversations.** IDs are local to a trace. In practice the tool
  definitions and system prompt are shared across sessions; replays model that as an
  assumption: kv-cache-tester's `--warm-prefix-pct` pre-warms part of that prefix, while
  AgentX inserts per-replay cache-bust markers so that unrelated replays do not share
  prefixes.
- **Absolute time.** Each trace starts at `t = 0`. Arrivals across sessions must be
  generated, and that schedule is an assumption.

## Mapping to the state-bench format

- `hash_scheme`: `weka-local-v1`. Each block hash is derived from the trace ID and the
  local block ID, so equal IDs map to equal hashes within a trace and never across
  traces. The partial last block of a prompt gets a hash unique to its request.
- `parent_req_id` and `pattern`: derived from the longest shared block prefix with an
  earlier request of the same agent. A request that contains all of its predecessor's
  full blocks is `append`; one that keeps only a prefix is `rewrite`; the first request
  of a subagent is `fork`.
- `segments`: `null`, because boundaries are incomplete. The tool and system boundary is
  kept in the session's import parameters.
- Labels: block structure and per-trace timing are recorded data; session start times
  and any warm-prefix sharing are assumptions.

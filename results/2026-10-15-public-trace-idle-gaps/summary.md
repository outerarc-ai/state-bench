# Idle gaps in public agent traces

**Result: in the 739 kv-cache-tester traces, most gaps between consecutive requests of a
chain are short (median 10 s), but most of the time that context sits idle falls in long
gaps. Weighted by context size, 97% of idle token-seconds are in gaps of 30 s or more and
85% in gaps of 10 min or more. Over the time a chain's context is held, it is idle 95% of
the time.** These are properties of the recorded traces (Claude Code sessions captured by
WEKA); no engine was run. Configuration: [config.yaml](config.yaml); data:
[raw/idle-gaps.json](raw/idle-gaps.json).

## Method

A gap is the client delay between the end of a request's response and the arrival of the
next request that extends it: `think_time` in the source, `tool_wait_s` after
`state-bench import weka`. During a gap an engine holds the chain's KV cache and, for a
hybrid model, its recurrent state, without using them. The context held is approximated
by the earlier request's input plus output tokens. "Idle token-seconds" weights each gap
by that context, which is proportional to the bytes held. Busy time is the source's
`api_time` (server time) of the earlier request.

The traces were converted with the new importer (`state-bench import weka`); parent links
come from shared block prefixes, as described in [Public agent traces](../../docs/PUBLIC_TRACES.md).

## Results

All values are derived from recorded trace data.

| Group | Gaps | Gap p50 / p90 / p99 | Context at gap, p50 | Idle token-s in gaps ≥ 30 s / ≥ 10 min | Idle share of held time |
| --- | --- | --- | --- | --- | --- |
| All | 58,894 | 10 s / 153 s / 51 min | 109k tokens | 97% / 85% | 95% |
| After a tool call (`stop = tool_use`) | 51,669 | 9 s / 65 s / 10 min | 108k | 83% / 44% | 81% |
| After a final answer (`stop = end_turn`) | 2,890 | 4.0 min / 64 min / 13 h | 126k | 100% / 96% | 99% |
| Stop reason not recorded (streaming) | 4,335 | 98 s / 19 min / 84 min | 110k | 99% / 74% | n/a¹ |
| Subagents | 663 | 4 s / 28 s / 75 s | 44k | 54% / 0% | 76% |

¹ Only 18 of these gaps have `api_time`; the overall idle share uses the 54,576 gaps that do.

Share of gaps and of idle token-seconds by gap length (all gaps):

| Gap | < 1 s | 1–5 s | 5–30 s | 30 s–2 min | 2–10 min | 10–60 min | ≥ 1 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Share of gaps | 0.3% | 14.1% | 60.4% | 12.9% | 8.3% | 3.4% | 0.7% |
| Share of idle token-seconds | 0.0% | 0.1% | 3.0% | 3.3% | 8.8% | 22.4% | 62.4% |

Other properties of the corpus:

- 59,937 requests: 59,204 from main agents and 733 from 70 subagents. Derived patterns:
  55,169 append, 3,911 rewrite, 70 fork, 787 roots (739 of them first requests).
- New input per append request: p50 666, p90 3,462, p99 16,235 tokens.
- 1,043 requests have no successor in their chain (session ends and abandoned branches);
  their median context is 125k tokens.

## Reading

- Holding every idle chain in HBM spends most of the capacity on chains that will not be
  used for minutes. A tier below HBM matters for the long tail of gaps, which is where
  almost all of the idle bytes-time is.
- Most gaps are tool calls of a few seconds. For these, the question is whether a
  restore can finish within the gap, or be hidden by fetching ahead of the next request.
  The [restore-versus-recompute estimate](../2026-10-15-restore-vs-recompute-estimate/summary.md)
  puts this in time.
- Subagent chains are shorter and busier. They gain little from a lower tier.

## Limits

- One corpus of one agent, Claude Code, using Anthropic models. Gaps include tool
  execution, human think time, and client overhead. Whether other agents behave alike
  has not been checked. The AgentX v1.0 build was not analyzed because the workspace
  cannot reach Hugging Face.
- The source records `think_time` in whole seconds.
- Each trace is one user's session, timed from its own start. Nothing here describes how
  many chains are idle at once on a server; that depends on the arrival schedule.
- The held context is approximate. The traces do not record whether the reply reappears
  token for token in the next request.

## Reproduce

```bash
results/2026-10-15-public-trace-idle-gaps/run.sh
```

`run.sh` downloads the traces at kv-cache-tester commit `94f6046` (about 630 MB) into a
temporary directory and runs [analyze.py](analyze.py). The corpus is Apache-2.0 and is
not copied into this repository.

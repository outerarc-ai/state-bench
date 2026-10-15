"""Idle gaps between consecutive requests of a conversation chain in WEKA-format traces.

Usage: python analyze.py TRACE_DIR_OR_FILE... > raw/idle-gaps.json

A gap is the client delay between the end of a request's response and the arrival of the
next request that extends it (state-bench ``tool_wait_s``; the source's ``think_time``).
The source records it in whole seconds. During a gap the chain's KV cache and model
state are held but unused. Context held is
approximated by the earlier request's input plus output tokens. Everything here is a
property of the recorded traces; no engine was run.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from state_bench.traces import weka

EDGES = [0, 1, 5, 30, 120, 600, 3600, float("inf")]
LABELS = ["<1s", "1-5s", "5-30s", "30s-2m", "2-10m", "10-60m", ">=1h"]
THRESHOLDS = [5, 30, 120, 600]


def bucket(g: float) -> int:
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= g < EDGES[i + 1]:
            return i
    return len(LABELS) - 1


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(gaps: list[tuple[float, int, float | None]]) -> dict:
    """gaps: (gap seconds, context tokens, server seconds of the earlier request or None)."""
    n = len(gaps)
    idle = sum(c * g for g, c, _ in gaps)
    timed = [(g, c, a) for g, c, a in gaps if a is not None]
    t_idle = sum(c * g for g, c, _ in timed)
    t_busy = sum(c * a for _, c, a in timed)
    counts = [0] * len(LABELS)
    weight = [0.0] * len(LABELS)
    for g, c, _ in gaps:
        b = bucket(g)
        counts[b] += 1
        weight[b] += c * g
    return {
        "gaps": n,
        "gap_s": {
            q: round(pct([g for g, _, _ in gaps], p), 2)
            for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        },
        "context_tokens": {
            q: int(pct([c for _, c, _ in gaps], p)) for q, p in (("p50", 0.5), ("p90", 0.9))
        },
        "share_of_gaps": dict(zip(LABELS, [round(x / n, 4) for x in counts], strict=True)),
        "share_of_idle_token_seconds": dict(
            zip(LABELS, [round(x / idle, 4) for x in weight], strict=True)
        ),
        "idle_token_seconds_at_or_above": {
            f">={t}s": round(sum(c * g for g, c, _ in gaps if g >= t) / idle, 4) for t in THRESHOLDS
        },
        # Over gaps whose earlier request records api_time (server time).
        "idle_share_of_held_token_seconds": (
            round(t_idle / (t_idle + t_busy), 4) if timed else None
        ),
        "gaps_with_api_time": len(timed),
    }


def main(paths: list[str]) -> None:
    groups: dict[str, list[tuple[float, int, float | None]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    ends: list[int] = []
    deltas: list[int] = []
    no_api = 0
    for origin, raw_trace in weka.read_weka(paths):
        raw: dict[str, dict] = {}
        _, reqs, _ = weka.convert_trace(raw_trace, origin, "x", "x", raw_out=raw)
        counts["sessions"] += 1
        idx = {r.req_id: r for r in reqs}
        for r in reqs:
            if r.pattern == "append":
                deltas.append(r.prompt_tokens - idx[r.parent_req_id].prompt_tokens)
            counts["requests"] += 1
            counts[f"pattern_{r.pattern or 'root'}"] += 1
            counts["subagent_requests" if "agent" in r.req_id else "main_requests"] += 1
            ctx = r.prompt_tokens + r.output_tokens
            if r.tool_wait_s is None:
                ends.append(ctx)
                continue
            src = raw[r.req_id]
            if "api_time" not in src:
                no_api += 1
            api = float(src["api_time"]) if "api_time" in src else None
            stop = src.get("stop") or "not_recorded"
            kind = stop if stop in ("tool_use", "end_turn") else "not_recorded"
            agent = "subagent" if "agent" in r.req_id else "main"
            item = (r.tool_wait_s, ctx, api)
            groups["all"].append(item)
            groups[f"stop={kind}"].append(item)
            groups[f"agent={agent}"].append(item)
    out = {
        "provenance": "recorded trace properties; derived gaps; no engine measurement",
        "counts": dict(sorted(counts.items())),
        "gaps_without_api_time": no_api,
        "chain_ends": {"count": len(ends), "context_tokens_p50": int(pct(ends, 0.5))},
        # New input tokens per append request: what a woken chain prefills on top of its cache.
        "append_new_tokens": {
            q: int(pct(deltas, p)) for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        },
        "groups": {k: summarize(v) for k, v in sorted(groups.items())},
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])

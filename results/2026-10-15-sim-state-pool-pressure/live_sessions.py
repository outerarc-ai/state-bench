"""Sessions live at each request's arrival in a trace, as the state pool sees them.

Usage: python live_sessions.py TRACE.jsonl > raw/subset-N-live.json

A session is live from its first request's arrival to its last request's arrival. For
each request, count the live sessions at its arrival; report the request-weighted mean
and percentiles. Dividing the state-pool size by this count gives the slots per live
session the simulator's pool had to share. Derived from the trace; no engine was run.
"""

from __future__ import annotations

import bisect
import json
import sys

from state_bench.traces.schema import load_trace


def main(path: str) -> None:
    trace = load_trace(path, check=False)
    arrivals = trace.global_arrivals()
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for t, r in arrivals:
        first.setdefault(r.session_id, t)
        last[r.session_id] = t
    starts = sorted(first.values())
    ends = sorted(last.values())
    live = sorted(bisect.bisect_right(starts, t) - bisect.bisect_left(ends, t) for t, _ in arrivals)
    n = len(live)
    out = {
        "provenance": "derived from the trace",
        "sessions": len(first),
        "requests": n,
        "live_sessions_mean": round(sum(live) / n, 2),
        "live_sessions_p10": live[n // 10],
        "live_sessions_p50": live[n // 2],
        "live_sessions_p90": live[n * 9 // 10],
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

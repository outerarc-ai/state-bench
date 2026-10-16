"""Summarize the retention ablation: placement held fixed, eviction varied.

Usage: python analyze.py RESULT_DIR > raw/summary.json

For each case and pool size, reports the recomputed share for each policy and the cut
from value-based eviction at fixed placement:
``boundary_value:eviction=lru`` versus ``boundary_value``, and ``prompt_end_anchor`` versus
``boundary_value:reply_end=false`` (both place at prompt ends only). Simulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cut(a: float, b: float) -> float:
    return round((a - b) / a, 4) if a else 0.0


def main(d: str) -> None:
    raw = Path(d) / "raw"
    cases = []
    for s in (0, 1, 2):
        for v in ("roundtrip", "no-roundtrip"):
            t = {
                (r["policy"], r["config"]["state_slots"]): r
                for r in json.load(open(raw / f"subset-{s}-{v}.json"))
            }
            slots = [*sorted({k[1] for k in t if k[1] is not None}, reverse=True), None]
            rows = []
            for n in slots:
                ratio = {p: t[(p, n)]["recompute_ratio"] for p, m in t if m == n}
                writes = {p: t[(p, n)]["snapshots_written"] for p, m in t if m == n}
                rows.append(
                    {
                        "slots": n,
                        "recompute_ratio": ratio,
                        "snapshots_written": writes,
                        "value_eviction_cut_with_reply_end": cut(
                            ratio["boundary_value:eviction=lru"], ratio["boundary_value"]
                        ),
                        "value_eviction_cut_prompt_end_only": cut(
                            ratio["prompt_end_anchor"], ratio["boundary_value:reply_end=false"]
                        ),
                    }
                )
            cases.append({"subset": s + 1, "roundtrip": v == "roundtrip", "rows": rows})
    json.dump({"provenance": "simulated", "cases": cases}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

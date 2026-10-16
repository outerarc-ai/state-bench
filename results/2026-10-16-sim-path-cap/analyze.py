"""Summarize the per-path cap sweep.

Usage: python analyze.py RESULT_DIR > raw/summary.json

For each case and pool size: the recomputed share and snapshots written per request for
``sglang_current`` at each per-path cap, and for the reference policies run without a
cap. Simulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CAPS = ["none", "1", "2", "4", "8"]


def main(d: str) -> None:
    raw = Path(d) / "raw"
    cases = []
    for s in (0, 1, 2):
        for v in ("roundtrip", "no-roundtrip"):
            rows: dict[object, dict] = {}
            for cap in CAPS:
                for r in json.load(open(raw / f"subset-{s}-{v}-cap-{cap}.json")):
                    n = r["config"]["state_slots"]
                    name = (
                        r["policy"]
                        if r["policy"] != "sglang_current"
                        else f"sglang_current@cap={cap}"
                    )
                    row = rows.setdefault(
                        n, {"slots": n, "recompute_ratio": {}, "writes_per_request": {}}
                    )
                    row["recompute_ratio"][name] = r["recompute_ratio"]
                    row["writes_per_request"][name] = round(
                        r["snapshots_written"] / r["requests"], 3
                    )
            order = [*sorted(k for k in rows if k is not None), None]
            cases.append(
                {"subset": s + 1, "roundtrip": v == "roundtrip", "rows": [rows[k] for k in order]}
            )
    json.dump({"provenance": "simulated", "cases": cases}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

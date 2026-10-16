"""Compare the pressure-gated per-path cap with fixed caps and value-based retention.

Usage: python analyze.py RESULT_DIR > raw/summary.json

Reads this directory's runs and, for the fixed-cap and reference rows, the runs of
``../2026-10-16-sim-path-cap`` (same traces and simulator settings). Simulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GATED = [("1", "0.8"), ("1", "0.95"), ("2", "0.8"), ("2", "0.95")]
FIXED = ["none", "1", "2", "4", "8"]


def main(d: str) -> None:
    here = Path(d)
    fixed_dir = here.parent / "2026-10-16-sim-path-cap" / "raw"
    cases = []
    for s in (0, 1, 2):
        for v in ("roundtrip", "no-roundtrip"):
            rows: dict[object, dict[str, float]] = {}
            for cap in FIXED:
                for r in json.load(open(fixed_dir / f"subset-{s}-{v}-cap-{cap}.json")):
                    name = r["policy"] if r["policy"] != "sglang_current" else f"cap={cap}"
                    rows.setdefault(r["config"]["state_slots"], {})[name] = r["recompute_ratio"]
            for cap, usage in GATED:
                f = here / "raw" / f"subset-{s}-{v}-cap-{cap}-usage-{usage}.json"
                for r in json.load(open(f)):
                    rows[r["config"]["state_slots"]][f"cap={cap}@{usage}"] = r["recompute_ratio"]
            order = [*sorted(k for k in rows if k is not None), None]
            out = []
            for n in order:
                row = rows[n]
                best_fixed = min(row[f"cap={c}"] for c in FIXED)
                out.append(
                    {
                        "slots": n,
                        "recompute_ratio": row,
                        "best_fixed_cap": best_fixed,
                        "gated_minus_best_fixed_pp": {
                            f"cap={c}@{u}": round(100 * (row[f"cap={c}@{u}"] - best_fixed), 3)
                            for c, u in GATED
                        },
                    }
                )
            cases.append({"subset": s + 1, "roundtrip": v == "roundtrip", "rows": out})
    json.dump({"provenance": "simulated", "cases": cases}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

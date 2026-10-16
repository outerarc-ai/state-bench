"""Summarize the pool sweep per state slot per live session.

Usage: python analyze.py RESULT_DIR > raw/summary.json

For each case (subset, reply round-trip) and pool size, report the recomputed share for
the best of the existing online policies (sglang_current, prompt_end_anchor,
marconi_admission, pin_recent), for boundary_value (the lower of its two settings), and
for the offline reference, with the pool expressed as slots per live session (pool size
divided by the request-weighted mean number of live sessions). Simulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXISTING = ["sglang_current", "prompt_end_anchor", "marconi_admission", "pin_recent"]
VALUE = ["boundary_value", "boundary_value:demand=true"]


def main(d: str) -> None:
    raw = Path(d) / "raw"
    cases = []
    for s in (0, 1, 2):
        live = json.load(open(raw / f"subset-{s}-live.json"))["live_sessions_mean"]
        for v in ("roundtrip", "no-roundtrip"):
            f = raw / f"subset-{s}-{v}.json"
            if not f.exists() or f.stat().st_size == 0:
                continue
            t: dict[tuple[str, object], float] = {}
            for r in json.load(open(f)):
                t[(r["policy"], r["config"]["state_slots"])] = r["recompute_ratio"]
            slots = [*sorted({k[1] for k in t if k[1] is not None}), None]
            rows = []
            for n in slots:
                best = min(EXISTING, key=lambda p: t[(p, n)])
                b = t[(best, n)]
                dv = min(t[(p, n)] for p in VALUE)
                o = t[("offline_oracle", n)]
                rows.append(
                    {
                        "slots": n,
                        "slots_per_live_session": None if n is None else round(n / live, 2),
                        "best_existing": best,
                        "best_existing_ratio": b,
                        "sglang_current_ratio": t[("sglang_current", n)],
                        "boundary_value_ratio": dv,
                        "offline_oracle_ratio": o,
                        "boundary_value_cut_vs_best": round((b - dv) / b, 4),
                        "best_existing_gap_to_oracle_pp": round(100 * (b - o), 3),
                    }
                )
            cases.append(
                {
                    "subset": s + 1,
                    "roundtrip": v == "roundtrip",
                    "live_sessions_mean": live,
                    "rows": rows,
                }
            )
    json.dump({"provenance": "simulated", "cases": cases}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

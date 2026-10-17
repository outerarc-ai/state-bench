"""Per-request cached tokens of the paired rehearsal.

Usage: python3 analyze.py   (reads raw/run-*.jsonl and raw/runs.json)

Checks that each arm's server-reported cached tokens are identical across its five runs,
and lists the requests whose cached tokens differ between arms. Values are measured
(reported by the server).
"""

from __future__ import annotations

import json
from pathlib import Path

RAW = Path(__file__).parent / "raw"


def main() -> None:
    manifest = json.loads((RAW / "runs.json").read_text())
    by_arm: dict[str, list[list[dict]]] = {}
    for entry in manifest:
        recs = [json.loads(line) for line in (RAW / entry["records"]).open()]
        by_arm.setdefault(entry["arm"], []).append(recs)
    cached = {
        arm: [tuple(r["cached_tokens"] for r in recs) for recs in runs]
        for arm, runs in by_arm.items()
    }
    for arm, seqs in cached.items():
        print(f"{arm}: cached tokens identical across {len(seqs)} runs: {len(set(seqs)) == 1}")
    ref = by_arm["baseline"][0]
    base, cand = cached["baseline"][0], cached["candidate"][0]
    prompt = [r["prompt_tokens"] for r in ref]
    print(f"prompt tokens {sum(prompt)}; cached baseline {sum(base)}, candidate {sum(cand)}")
    print("requests that differ (req_id, prompt, baseline cached, candidate cached):")
    for r, p, b, c in zip(ref, prompt, base, cand, strict=True):
        if b != c:
            print(f"  {r['req_id']:6s} {p:5d} {b:5d} {c:5d}")


if __name__ == "__main__":
    main()

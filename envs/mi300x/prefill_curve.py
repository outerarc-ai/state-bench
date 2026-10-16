"""Prefill time against prompt length on a running SGLang server.

Usage: python3 prefill_curve.py URL OUT.json [--lengths 1024,4096,...] [--reps 5]

For each length, sends `reps` prompts of fresh random token ids with one output token,
one request at a time, after flushing the server's prefix cache, and records the
end-to-end request time. With one output token this is close to the prefill time plus
fixed per-request overhead; the shortest length shows that overhead. Times are measured
on whatever device the server runs; report the device, image, and server arguments with
the result.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, body: dict | None, timeout: float = 1800) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        text = r.read().decode()
    return json.loads(text) if text.strip().startswith("{") else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768,65536")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    lengths = [int(x) for x in a.lengths.split(",")]
    rng = random.Random(a.seed)
    rows = []
    post(base + "/generate", {"input_ids": [1000] * 64, "sampling_params": {"max_new_tokens": 1}})
    for n in lengths:
        times = []
        for rep in range(a.reps):
            post(base + "/flush_cache", None)
            ids = [rng.randrange(1000, 100_000) for _ in range(n)]
            t0 = time.perf_counter()
            r = post(
                base + "/generate",
                {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
            )
            dt = time.perf_counter() - t0
            cached = (r.get("meta_info") or {}).get("cached_tokens")
            times.append(dt)
            print(f"L {n} rep {rep} {dt * 1e3:.1f} ms cached {cached}", flush=True)
        med = statistics.median(times)
        rows.append(
            {
                "prompt_tokens": n,
                "times_s": times,
                "median_s": med,
                "median_us_per_token": med / n * 1e6,
            }
        )
    out = {
        "provenance": "measured",
        "method": __doc__.split("\n\n")[2].replace("\n", " "),
        "url": a.url,
        "reps": a.reps,
        "seed": a.seed,
        "rows": rows,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    for r in rows:
        print(
            f"L {r['prompt_tokens']:6d}  median {r['median_s'] * 1e3:9.1f} ms  "
            f"{r['median_us_per_token']:7.2f} us/token"
        )


if __name__ == "__main__":
    main()

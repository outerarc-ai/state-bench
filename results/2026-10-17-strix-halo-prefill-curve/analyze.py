"""Summarize the prefill-curve run and compare its shape with a FLOP model.

Usage: python3 analyze.py   (reads raw/round17-partA.log, writes data.json)

Times are measured on a shared gfx1151 (development GPU); they show the shape of the
curve, not MI300X performance. The FLOP model uses the model's published configuration
(spec) and a parameter count derived from measured weight memory (see
../2026-10-15-restore-vs-recompute-estimate/inputs.json).
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
PARAMS = 8.84e9  # derived: measured bf16 weight memory on gfx1151 / 2 bytes
ATTN_LAYERS, Q_HEADS, HEAD_DIM = 4, 40, 128  # spec: config.json (checked in this run's log)


def flops(n: int) -> float:
    """Prefill FLOPs for an n-token prompt: 2*params per token plus causal attention
    (QK^T and PV, 2 FLOPs per multiply-add) summed over positions."""
    linear = 2 * PARAMS * n
    attention = ATTN_LAYERS * 2 * 2 * Q_HEADS * HEAD_DIM * n * (n + 1) / 2
    return linear + attention


def solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a 3x3 linear system by Gaussian elimination."""
    m = [[*row, v] for row, v in zip(a, b, strict=True)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(3):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i], strict=True)]
    return [m[i][3] / m[i][i] for i in range(3)]


def main() -> None:
    log = (HERE / "raw" / "round17-partA.log").read_text()
    runs: dict[int, list[float]] = {}
    for n, ms in re.findall(r"^rep \d+ L (\d+) time ([\d.]+) ms cached 0$", log, re.M):
        runs.setdefault(int(n), []).append(float(ms) / 1e3)
    lengths = sorted(runs)
    med = {n: statistics.median(runs[n]) for n in lengths}
    # Least-squares fit t = c + a*n + b*n^2 on the medians.
    xs = [[1.0, n, n * n] for n in lengths]
    ata = [[sum(r[i] * r[j] for r in xs) for j in range(3)] for i in range(3)]
    atb = [sum(r[i] * med[n] for r, n in zip(xs, lengths, strict=True)) for i in range(3)]
    c, a, b = solve3(ata, atb)
    rows = []
    for n in lengths:
        fit = c + a * n + b * n * n
        rows.append(
            {
                "prompt_tokens": n,
                "times_s": runs[n],
                "median_s": med[n],
                "us_per_token": med[n] / n * 1e6,
                "fit_s": fit,
                "model_tflops": flops(n) / 1e12,
                "effective_tflops": flops(n) / med[n] / 1e12,
            }
        )
    first, last = lengths[0], lengths[-1]
    out = {
        "provenance": {
            "times": "measured (shared gfx1151, end-to-end request time, 1 output token, "
            "prefix cache flushed before each request)",
            "flops": "derived from spec config and a derived parameter count",
            "fit": "derived",
        },
        "rows": rows,
        "fit": {"c_s": c, "a_s_per_token": a, "b_s_per_token2": b},
        "per_token_growth": {
            "measured": (med[last] / last) / (med[first] / first),
            "flop_model": (flops(last) / last) / (flops(first) / first),
        },
    }
    (HERE / "data.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'L':>6} {'median s':>9} {'us/tok':>8} {'fit s':>8} {'eff TFLOPS':>10}")
    for r in rows:
        print(
            f"{r['prompt_tokens']:6d} {r['median_s']:9.3f} {r['us_per_token']:8.1f} "
            f"{r['fit_s']:8.3f} {r['effective_tflops']:10.2f}"
        )
    g = out["per_token_growth"]
    print(f"fit: c={c:.3f} s, a={a * 1e6:.1f} us/token, b={b * 1e12:.2f} ps/token^2")
    print(
        f"per-token time {last} vs {first}: measured x{g['measured']:.3f}, "
        f"FLOP model x{g['flop_model']:.3f}"
    )


if __name__ == "__main__":
    main()

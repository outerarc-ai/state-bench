"""Restore-versus-recompute estimate for waking an idle agent chain on one MI300X.

Usage: python estimate.py inputs.json > raw/estimate.json

Saving a chain's cache to a tier and restoring it beats evicting and recomputing when

    (s_kv * L + S_state) / B_tier + t_fixed < T_prefill(L)

T_prefill(L) = (2 * P * L + A * L^2) / (peak * mfu), where A * L^2 counts the causal
attention FLOPs (A = 2 * attention_layers * q_heads * head_dim). Time to first token after
waking with ``new`` tokens appended:

    keep in HBM:  T_inc(L, new)
    restore:      (s_kv * L + S_state) / B_tier + t_fixed + T_inc(L, new)
    recompute:    T_prefill(L + new)

with T_inc(L, new) = (2 * P * new + 2 * A * new * (L + new / 2)) / (peak * mfu).
Decode of the first token, queueing, and interference with other requests are ignored.
Every input carries its source in inputs.json; outputs are estimates, not measurements.
"""

from __future__ import annotations

import json
import sys


def flops_prefill(m: dict, n: float) -> float:
    a = 2 * m["attention_layers"] * m["q_heads"] * m["head_dim"]
    return 2 * m["params"] * n + a * n * n


def flops_incremental(m: dict, ctx: float, new: float) -> float:
    a = 2 * m["attention_layers"] * m["q_heads"] * m["head_dim"]
    return 2 * m["params"] * new + 2 * a * new * (ctx + new / 2)


def restore_s(m: dict, tier: dict, ctx: float) -> float:
    return (m["kv_bytes_per_token"] * ctx + m["state_bytes"]) / tier["bandwidth_Bps"] + tier[
        "fixed_s"
    ]


def crossover(m: dict, tier: dict, rate: float) -> int:
    """Smallest prefix length L at which restoring from ``tier`` beats recomputing."""
    lo, hi = 1, 1
    while restore_s(m, tier, hi) >= flops_prefill(m, hi) / rate:
        hi *= 2
        if hi > 1 << 30:
            return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if restore_s(m, tier, mid) < flops_prefill(m, mid) / rate:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main(path: str) -> None:
    cfg = json.load(open(path))
    m = {k: v["value"] for k, v in cfg["model"].items() if isinstance(v, dict)}
    gpu = {k: v["value"] for k, v in cfg["gpu"].items() if isinstance(v, dict)}
    tiers = {name: {k: v["value"] for k, v in t.items()} for name, t in cfg["tiers"].items()}
    out: dict = {"provenance": "estimate from spec values and assumptions; not measured"}
    for mfu in gpu["mfu"]:
        rate = gpu["peak_flops"] * mfu
        key = f"mfu={mfu}"
        res: dict = {"recompute_us_per_token_short_prefix": round(2 * m["params"] / rate * 1e6, 2)}
        res["crossover_tokens"] = {name: crossover(m, t, rate) for name, t in tiers.items()}
        rows = []
        for ctx in cfg["contexts"]:
            for new in cfg["new_tokens"]:
                inc = flops_incremental(m, ctx["tokens"], new["tokens"]) / rate
                row = {
                    "context": ctx["label"],
                    "context_tokens": ctx["tokens"],
                    "new": new["label"],
                    "new_tokens": new["tokens"],
                    "ttft_ms": {"keep_in_hbm": round(inc * 1e3, 1)},
                    "held_GB": round(
                        (m["kv_bytes_per_token"] * ctx["tokens"] + m["state_bytes"]) / 1e9, 3
                    ),
                }
                for name, t in tiers.items():
                    row["ttft_ms"][f"restore_from_{name}"] = round(
                        (restore_s(m, t, ctx["tokens"]) + inc) * 1e3, 1
                    )
                row["ttft_ms"]["recompute"] = round(
                    flops_prefill(m, ctx["tokens"] + new["tokens"]) / rate * 1e3, 1
                )
                rows.append(row)
        res["wake_ttft"] = rows
        out[key] = res
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

"""State slots per KV-resident session under SGLang's default memory split.

Usage: python deployment.py models.json > raw/deployment.json

When ``--max-mamba-cache-size`` is not set, SGLang splits the memory left after weights
between recurrent state and KV in the ratio ``mamba_full_memory_ratio`` : 1 (default 0.9;
SGLang c4870d98 ``mem_cache/kv_cache_configurator.py:3016-3028`` and
``arg_groups/fields/schedule.py:216-222``).
If every live session keeps its whole context of ``ctx`` tokens in KV, the number of state
slots per session is independent of GPU memory:

    slots per session = ratio * ctx * kv_bytes_per_token / state_bytes_per_slot

Sizes are per GPU at the given tensor-parallel degree: recurrent heads and KV heads are
split across GPUs, and a KV head is replicated when TP exceeds the number of KV heads.
Recurrent state is fp32 (``mamba_ssm_dtype``), convolution state and KV are bf16. Each
running request also holds 5 slots (3, plus 2 tracking slots with the extra buffer and
the overlap scheduler; ``mem_cache/kv_cache_configurator.py:222,2740-2766`` and
``mem_cache/allocation_sizing.py:11-19``). Agent sessions are idle most of the time, so
this is small; it is reported separately.
"""

from __future__ import annotations

import json
import math
import sys

RATIO = 0.9
CONTEXTS = [10_000, 30_000, 109_143, 297_200]  # last two: p50 and p90 context at an idle gap
RUNNING_SLOTS = 5


def sizes(m: dict, tp: int) -> tuple[int, int]:
    kv_heads_per_gpu = max(1, m["kv_heads"] // tp)
    kv = m["attention_layers"] * 2 * kv_heads_per_gpu * m["head_dim"] * 2
    k = m["conv_kernel"] - 1
    if m["kind"] == "mamba2":
        heads = m["mamba_heads"] // tp
        ssm = heads * m["mamba_head_dim"] * m["ssm_state"] * 4
        conv = (heads * m["mamba_head_dim"] + 2 * m["n_groups"] * m["ssm_state"] // tp) * k * 2
    else:
        ssm = (m["v_heads"] // tp) * m["k_dim"] * m["v_dim"] * 4
        conv_dim = 2 * (m["k_heads"] // tp) * m["k_dim"] + (m["v_heads"] // tp) * m["v_dim"]
        conv = conv_dim * k * 2
    return kv, m["recurrent_layers"] * (ssm + conv)


def main(path: str) -> None:
    spec = json.load(open(path))
    rows = []
    for m in spec["models"]:
        for tp in m["tp"]:
            kv, state = sizes(m, tp)
            row = {
                "model": m["name"],
                "tp": tp,
                "kv_bytes_per_token_per_gpu": kv,
                "state_bytes_per_slot_per_gpu": state,
                "state_slot_in_kv_tokens": round(state / kv),
                "slots_per_session": {str(c): round(RATIO * c * kv / state, 1) for c in CONTEXTS},
                "context_for_one_slot_per_session": math.ceil(state / (RATIO * kv)),
            }
            rows.append(row)
    out = {
        "provenance": "derived from published model configs and SGLang's default split",
        "mamba_full_memory_ratio": RATIO,
        "running_request_slots": RUNNING_SLOTS,
        "models": rows,
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main(sys.argv[1])

"""A deterministic multi-turn load for SGLang's native ``/generate`` endpoint.

Sessions share a common prefix (a stand-in system prompt) and grow as append chains:
each turn adds new tokens to the previous prompt. Requests are sent one at a time in
round-robin order across sessions, so sessions interleave and compete for the engine's
caches. Token ids are placeholders from a seeded generator; the reply is not appended,
which matches an engine whose chat template drops earlier replies.

Each request is streamed so that time to first token is measured, and the server's
``meta_info`` supplies output and cached-token counts. Records follow
:mod:`state_bench.bench.records`. Times are measured; the workload shape is an
assumption.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadShape:
    sessions: int = 8
    turns: int = 6
    shared_prefix: int = 1024
    turn_min: int = 256
    turn_max: int = 1024
    max_new_tokens: int = 16
    vocab_low: int = 1000
    vocab_high: int = 100_000
    seed: int = 17

    def __post_init__(self) -> None:
        if self.sessions < 1 or self.turns < 1:
            raise ValueError("sessions and turns must be >= 1")
        if not 0 < self.turn_min <= self.turn_max:
            raise ValueError("need 0 < turn_min <= turn_max")
        if not 0 <= self.vocab_low < self.vocab_high:
            raise ValueError("need 0 <= vocab_low < vocab_high")


def build_prompts(shape: LoadShape) -> list[tuple[str, list[int]]]:
    """``(req_id, input_ids)`` in send order. Deterministic for a given shape."""
    rng = random.Random(shape.seed)

    def tok(n: int) -> list[int]:
        return [rng.randrange(shape.vocab_low, shape.vocab_high) for _ in range(n)]

    shared = tok(shape.shared_prefix)
    adds = [
        [tok(rng.randint(shape.turn_min, shape.turn_max)) for _ in range(shape.turns)]
        for _ in range(shape.sessions)
    ]
    prompts = [list(shared) for _ in range(shape.sessions)]
    out = []
    for t in range(shape.turns):
        for s in range(shape.sessions):
            prompts[s] = prompts[s] + adds[s][t]
            out.append((f"s{s}-t{t}", list(prompts[s])))
    return out


def stream_generate(url: str, ids: list[int], max_new_tokens: int, timeout: float = 900):
    """Send one streamed request; return (ttft_s, e2e_s, last meta_info)."""
    body = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        "stream": True,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    # The server is local to the benchmark host; never route through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter()
    ttft = None
    meta: dict = {}
    with opener.open(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            meta = json.loads(payload).get("meta_info", meta)
    return ttft, time.perf_counter() - t0, meta


def run_load(
    url: str,
    out: str | Path,
    shape: LoadShape,
    *,
    send: Callable[[str, list[int], int], tuple] = stream_generate,
) -> int:
    """Send the load and write one record per request. Returns the number of failures."""
    start = time.perf_counter()
    failures = 0
    with open(out, "w", encoding="utf-8") as f:
        for req_id, ids in build_prompts(shape):
            rec: dict = {
                "req_id": req_id,
                "t_send_s": round(time.perf_counter() - start, 6),
                "prompt_tokens": len(ids),
                "valid": None,
            }
            try:
                ttft, e2e, meta = send(url, ids, shape.max_new_tokens)
                rec.update(
                    admitted=True,
                    completed=ttft is not None,
                    ttft_s=ttft,
                    e2e_s=e2e,
                    output_tokens=meta.get("completion_tokens"),
                    cached_tokens=meta.get("cached_tokens"),
                )
            except Exception as e:  # recorded, not raised: the run continues
                failures += 1
                rec.update(admitted=False, completed=False, error=f"{type(e).__name__}: {e}"[:300])
            f.write(json.dumps(rec) + "\n")
            f.flush()
    return failures

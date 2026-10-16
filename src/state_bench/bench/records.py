"""Per-request records of a load run, and the four denominators.

A load generator writes one JSON object per request to a JSONL file. Fields:

=================  ========================================================================
``req_id``         Request identifier, stable across runs of the same workload (required)
``t_send_s``       Send time relative to the start of the run (required)
``admitted``       The server accepted the request (no rejection or connection error)
``completed``      The response finished
``valid``          Output passed the correctness check; ``null`` when not checked
``ttft_s``         Time to first token; ``null`` if none arrived
``e2e_s``          Time to the last token; ``null`` if the request did not complete
``prompt_tokens``  Input tokens, as reported by the server when available
``output_tokens``  Output tokens
``cached_tokens``  Prompt tokens served from cache, as reported by the server, else ``null``
``error``          Error text, or ``null``
=================  ========================================================================

Every record is an offered request. Denominators follow ``docs/MEASUREMENT.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from .stats import percentile


@dataclass(frozen=True)
class RequestResult:
    req_id: str
    t_send_s: float
    admitted: bool = True
    completed: bool = True
    valid: bool | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None


_FIELDS = {f.name for f in fields(RequestResult)}


def load_records(path: str | Path) -> list[RequestResult]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            unknown = set(d) - _FIELDS
            if unknown:
                raise ValueError(f"{path}:{i}: unknown fields {sorted(unknown)}")
            if "req_id" not in d or "t_send_s" not in d:
                raise ValueError(f"{path}:{i}: req_id and t_send_s are required")
            out.append(RequestResult(**d))
    return out


def drop_warmup(records: Sequence[RequestResult], n: int) -> list[RequestResult]:
    """Drop the first ``n`` requests by send time."""
    return sorted(records, key=lambda r: (r.t_send_s, r.req_id))[n:]


@dataclass(frozen=True)
class Slo:
    ttft_s: float | None = None
    e2e_s: float | None = None

    def met(self, r: RequestResult) -> bool:
        if self.ttft_s is not None and (r.ttft_s is None or r.ttft_s > self.ttft_s):
            return False
        return not (self.e2e_s is not None and (r.e2e_s is None or r.e2e_s > self.e2e_s))


def denominators(records: Iterable[RequestResult], slo: Slo) -> dict:
    """Offered, admitted, completed, and on-time valid counts.

    ``on_time_valid`` counts completed requests within the SLO whose output was checked
    and found valid. It is ``None`` when no request's output was checked; ``on_time``
    gives the count without the validity condition.
    """
    rs = list(records)
    completed = [r for r in rs if r.admitted and r.completed]
    on_time = [r for r in completed if slo.met(r)]
    checked = any(r.valid is not None for r in rs)
    return {
        "offered": len(rs),
        "admitted": sum(r.admitted for r in rs),
        "completed": len(completed),
        "on_time": len(on_time),
        "on_time_valid": sum(r.valid is True for r in on_time) if checked else None,
        "validity_checked": checked,
    }


def run_metrics(records: Sequence[RequestResult]) -> dict[str, float | None]:
    """Request-level summary metrics of one run, over completed requests."""
    done = [r for r in records if r.admitted and r.completed]
    ttft = [r.ttft_s for r in done if r.ttft_s is not None]
    e2e = [r.e2e_s for r in done if r.e2e_s is not None]
    prompt = sum(r.prompt_tokens or 0 for r in done)
    cached = [r.cached_tokens for r in done if r.cached_tokens is not None]
    out: dict[str, float | None] = {
        "ttft_p50_s": percentile(ttft, 50) if ttft else None,
        "ttft_p90_s": percentile(ttft, 90) if ttft else None,
        "e2e_p50_s": percentile(e2e, 50) if e2e else None,
        "e2e_p90_s": percentile(e2e, 90) if e2e else None,
        "cached_token_share": (
            sum(cached) / prompt if cached and len(cached) == len(done) and prompt else None
        ),
    }
    return out


def matched_ttft_diff(base: Sequence[RequestResult], cand: Sequence[RequestResult]) -> float | None:
    """Median per-request TTFT difference (candidate minus baseline) over requests that
    completed with a TTFT in both runs."""
    b = {r.req_id: r.ttft_s for r in base if r.completed and r.ttft_s is not None}
    diffs = [
        r.ttft_s - b[r.req_id]
        for r in cand
        if r.completed and r.ttft_s is not None and r.req_id in b
    ]
    if not diffs:
        return None
    return percentile(diffs, 50)

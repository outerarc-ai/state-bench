"""Paired statistics for baseline-versus-candidate runs.

Each pair is one baseline run and one candidate run made back to back. The sample is the
per-pair difference (candidate minus baseline); the report is its median with a 95%
percentile bootstrap interval, as ``docs/MEASUREMENT.md`` requires. The bootstrap uses a
seeded generator, so the same inputs always give the same interval.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

MIN_PAIRS = 5


@dataclass(frozen=True)
class PairedResult:
    metric: str
    pairs: int
    baseline_median: float
    candidate_median: float
    diff_median: float
    """Median of candidate minus baseline over pairs."""
    ci_low: float
    ci_high: float
    relative_diff_median: float | None
    """Median of (candidate - baseline) / baseline; ``None`` if any baseline is 0."""
    verdict: str
    """``candidate lower``, ``candidate higher``, ``no significant difference``, or
    ``too few pairs`` (fewer than :data:`MIN_PAIRS`)."""

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(xs[lo])
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_median_ci(
    diffs: Sequence[float], *, n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the median of ``diffs``."""
    if not diffs:
        raise ValueError("no differences to resample")
    rng = random.Random(seed)
    n = len(diffs)
    meds = sorted(statistics.median(rng.choices(diffs, k=n)) for _ in range(n_boot))
    tail = (1.0 - level) / 2.0 * 100.0
    return percentile(meds, tail), percentile(meds, 100.0 - tail)


def compare_pairs(
    metric: str,
    pairs: Sequence[tuple[float, float]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Compare ``(baseline, candidate)`` values of one metric over pairs of runs."""
    if not pairs:
        raise ValueError(f"{metric}: no pairs")
    base = [b for b, _ in pairs]
    cand = [c for _, c in pairs]
    diffs = [c - b for b, c in pairs]
    lo, hi = bootstrap_median_ci(diffs, n_boot=n_boot, seed=seed)
    rel = (
        None
        if any(b == 0 for b in base)
        else statistics.median(d / b for d, b in zip(diffs, base, strict=True))
    )
    if len(pairs) < MIN_PAIRS:
        verdict = "too few pairs"
    elif lo <= 0.0 <= hi:
        verdict = "no significant difference"
    elif hi < 0.0:
        verdict = "candidate lower"
    else:
        verdict = "candidate higher"
    return PairedResult(
        metric=metric,
        pairs=len(pairs),
        baseline_median=statistics.median(base),
        candidate_median=statistics.median(cand),
        diff_median=statistics.median(diffs),
        ci_low=lo,
        ci_high=hi,
        relative_diff_median=rel,
        verdict=verdict,
    )

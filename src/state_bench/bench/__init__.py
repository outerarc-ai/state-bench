"""Paired baseline/candidate measurement: run orchestration, records, and statistics."""

from .paired import ARMS, DEFAULT_METRICS, PlannedRun, plan, report, run
from .records import RequestResult, Slo, denominators, load_records, run_metrics
from .stats import MIN_PAIRS, PairedResult, bootstrap_median_ci, compare_pairs, percentile

__all__ = [
    "ARMS",
    "DEFAULT_METRICS",
    "MIN_PAIRS",
    "PairedResult",
    "PlannedRun",
    "RequestResult",
    "Slo",
    "bootstrap_median_ci",
    "compare_pairs",
    "denominators",
    "load_records",
    "percentile",
    "plan",
    "report",
    "run",
    "run_metrics",
]

"""Alternating baseline/candidate runs and their paired report.

``docs/MEASUREMENT.md`` asks for baseline and candidate runs that alternate on an
exclusive device, at least five pairs, a server restart between runs, and paired
statistics with all four denominators. This module orchestrates that with
engine-agnostic shell commands and reports on the result.

A run configuration is a JSON object::

    {
      "name": "hicache-prefetch",
      "pairs": 5,
      "order": "ab",                      # or "abba" (alternates which arm goes first)
      "warmup_requests": 8,               # dropped from each run's records
      "slo": {"ttft_s": 2.0, "e2e_s": null},
      "device_check": ["amd-smi", "process"],   # output recorded before and after each run
      "device_idle_pattern": null,        # a substring the check prints when the device is idle
      "ready": {"command": ["curl", "-sf", "http://127.0.0.1:30000/health"],
                "timeout_s": 900, "interval_s": 5},
      "load": ["python", "replay.py", "--out", "{records}"],
      "arms": {
        "baseline":  {"start": ["python", "-m", "sglang.launch_server", "..."], "env": {}},
        "candidate": {"start": ["..."], "stop": ["..."]}
      }
    }

``{records}``, ``{run_dir}``, ``{arm}``, and ``{run}`` in any command are replaced per
run. The load command writes one JSON record per request (see
:mod:`state_bench.bench.records`). Without a ``stop`` command the server's process group
is terminated. Commands run in order; this module never runs two servers at once.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .records import Slo, denominators, drop_warmup, load_records, matched_ttft_diff, run_metrics
from .stats import MIN_PAIRS, compare_pairs

ARMS = ("baseline", "candidate")
DEFAULT_METRICS = ("ttft_p50_s", "ttft_p90_s", "e2e_p50_s", "e2e_p90_s", "cached_token_share")


@dataclass(frozen=True)
class PlannedRun:
    run: int
    pair: int
    arm: str


def plan(pairs: int, order: str = "ab") -> list[PlannedRun]:
    """Run order for ``pairs`` pairs. ``ab``: baseline first in every pair. ``abba``:
    the first arm alternates between pairs, which cancels a linear drift."""
    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    if order not in ("ab", "abba"):
        raise ValueError("order must be 'ab' or 'abba'")
    out: list[PlannedRun] = []
    for p in range(pairs):
        arms = ARMS if order == "ab" or p % 2 == 0 else ARMS[::-1]
        for arm in arms:
            out.append(PlannedRun(run=len(out), pair=p, arm=arm))
    return out


def _fill(cmd: Sequence[str], values: dict[str, str]) -> list[str]:
    out = []
    for part in cmd:
        for k, v in values.items():
            part = part.replace("{" + k + "}", v)
        out.append(part)
    return out


def _now() -> str:
    # Run metadata only; traces and reports never depend on it.
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check(cmd: Sequence[str] | None) -> str | None:
    if not cmd:
        return None
    p = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    return (p.stdout + p.stderr).strip()


def _wait_ready(
    ready: dict[str, Any], server: subprocess.Popen, values: dict[str, str]
) -> float | None:
    """Seconds until the ready command succeeds, or ``None`` on timeout or server exit."""
    cmd = _fill(ready["command"], values)
    timeout = float(ready.get("timeout_s", 900))
    interval = float(ready.get("interval_s", 5))
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if server.poll() is not None:
            return None
        if subprocess.run(cmd, capture_output=True, check=False).returncode == 0:
            return round(time.monotonic() - t0, 3)
        time.sleep(interval)
    return None


def _stop(
    server: subprocess.Popen, stop: Sequence[str] | None, values: dict[str, str]
) -> int | None:
    if stop:
        subprocess.run(_fill(stop, values), check=False)
    if server.poll() is None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    return server.returncode


def run(config: dict[str, Any], run_dir: str | Path, *, log: Callable[[str], None] = print) -> Path:
    """Execute the planned runs; write ``raw/run-NN-<arm>.jsonl`` and ``raw/runs.json``."""
    run_dir = Path(run_dir)
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = []
    for pr in plan(int(config.get("pairs", MIN_PAIRS)), config.get("order", "ab")):
        arm = config["arms"][pr.arm]
        records = raw / f"run-{pr.run:02d}-{pr.arm}.jsonl"
        values = {
            "records": str(records),
            "run_dir": str(run_dir),
            "arm": pr.arm,
            "run": str(pr.run),
        }
        entry: dict[str, Any] = {
            "run": pr.run,
            "pair": pr.pair,
            "arm": pr.arm,
            "records": records.name,
            "device_before": _check(config.get("device_check")),
            "started_at": _now(),
        }
        log(f"run {pr.run} pair {pr.pair} {pr.arm}: starting server")
        env = {**os.environ, **{k: str(v) for k, v in arm.get("env", {}).items()}}
        server = subprocess.Popen(
            _fill(arm["start"], values),
            env=env,
            stdout=open(raw / f"run-{pr.run:02d}-{pr.arm}.server.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            entry["ready_after_s"] = _wait_ready(config["ready"], server, values)
            if entry["ready_after_s"] is None:
                entry["status"] = "server not ready"
            else:
                load = subprocess.run(_fill(config["load"], values), check=False)
                entry["load_exit"] = load.returncode
                ok = load.returncode == 0 and records.exists()
                entry["status"] = "ok" if ok else "load failed"
        finally:
            entry["server_exit"] = _stop(server, arm.get("stop"), values)
        entry["ended_at"] = _now()
        entry["device_after"] = _check(config.get("device_check"))
        log(f"run {pr.run} pair {pr.pair} {pr.arm}: {entry['status']}")
        manifest.append(entry)
        (raw / "runs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def report(run_dir: str | Path, *, metrics: Sequence[str] = DEFAULT_METRICS, seed: int = 0) -> dict:
    """Paired report of a run directory written by :func:`run` (or by hand, same layout)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "raw" / "runs.json").read_text())
    slo = Slo(**(config.get("slo") or {}))
    warm = int(config.get("warmup_requests", 0))
    idle = config.get("device_idle_pattern")
    runs: dict[int, dict] = {}
    warnings: list[str] = []
    for e in manifest:
        entry = {k: e.get(k) for k in ("run", "pair", "arm", "status")}
        if e.get("status") == "ok":
            recs = drop_warmup(load_records(run_dir / "raw" / e["records"]), warm)
            entry["denominators"] = denominators(recs, slo)
            entry["metrics"] = run_metrics(recs)
            entry["_records"] = recs
        else:
            warnings.append(f"run {e['run']} ({e['arm']}): {e.get('status')}")
        if idle is not None:
            for when in ("device_before", "device_after"):
                if e.get(when) is not None and idle not in e[when]:
                    warnings.append(f"run {e['run']} ({e['arm']}): device not idle ({when})")
        runs[e["run"]] = entry
    pairs: dict[int, dict[str, dict]] = {}
    for entry in runs.values():
        pairs.setdefault(entry["pair"], {})[entry["arm"]] = entry
    complete = {
        p: arms
        for p, arms in sorted(pairs.items())
        if all(a in arms and arms[a].get("status") == "ok" for a in ARMS)
    }
    if len(complete) < MIN_PAIRS:
        warnings.append(f"{len(complete)} complete pairs; the protocol needs at least {MIN_PAIRS}")
    comparisons = []
    for m in metrics:
        vals = [
            (arms["baseline"]["metrics"][m], arms["candidate"]["metrics"][m])
            for arms in complete.values()
            if arms["baseline"]["metrics"].get(m) is not None
            and arms["candidate"]["metrics"].get(m) is not None
        ]
        if vals:
            comparisons.append(compare_pairs(m, vals, seed=seed).to_dict())
    matched = [
        d
        for arms in complete.values()
        if (d := matched_ttft_diff(arms["baseline"]["_records"], arms["candidate"]["_records"]))
        is not None
    ]
    if matched:
        r = compare_pairs("matched_ttft_diff_s", [(0.0, d) for d in matched], seed=seed).to_dict()
        r["note"] = "per pair: median over requests of candidate minus baseline TTFT"
        comparisons.append(r)
    for entry in runs.values():
        entry.pop("_records", None)
    return {
        "provenance": "measured",
        "name": config.get("name"),
        "slo": config.get("slo"),
        "warmup_requests": warm,
        "complete_pairs": len(complete),
        "runs": [runs[k] for k in sorted(runs)],
        "comparisons": comparisons,
        "warnings": warnings,
    }

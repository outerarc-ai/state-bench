"""Import traces in the WEKA trace format (kv-cache-tester, AgentX).

The format stores one JSON object per trace: trace-level metadata and a list of requests,
each with arrival time ``t``, token counts ``in`` and ``out``, the integer IDs of the
prompt's full 64-token blocks (``hash_ids``, unique within one trace), and client timing.
A request of ``type: subagent`` nests a subagent's own trace. See
``docs/PUBLIC_TRACES.md`` for the field list and what the format does not record.

Mapping (hash scheme ``weka-local-v1``):

* One trace becomes one session; subagent requests join it. Request arrival times are
  recorded data. Session start times are an assumption (``session_interval_s``).
* Each local block ID maps to ``blake2b(trace id, block id)``, so equal IDs give equal
  hashes within a trace and never across traces. The partial last block of a prompt gets
  a hash unique to its request.
* Parent and pattern are derived per agent: the parent is the earlier request of the
  same agent with the longest shared full-block prefix (the latest one on a tie). The
  search relies on block IDs behaving like chained hashes, which the sampled traces
  show. A
  request that contains all of its parent's full blocks and is longer is ``append``; one
  with the same blocks and length is ``retry``; any other with at least one shared block
  is ``rewrite``. A request sharing no block with an earlier request of its agent is a
  root, except the first request of a subagent, which is a ``fork`` of the latest
  request of the spawning agent that arrived before it.
* ``append`` assumes the child also repeats the tokens of the parent's partial last
  block; the format does not record them.
* ``tool_wait_s`` of a request is the ``think_time`` of its earliest non-fork child. In
  a 60-trace kv-cache-tester sample, ``t`` of a request equals the previous request's
  ``t + api_time`` plus its own ``think_time`` to within 0.7 s for 90% of requests.
* ``segments`` is ``null``: the format records only the length of the tool and system
  prefix, kept per session in the header's ``params.prefix_tokens``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .schema import (
    BLOCK_TOKENS,
    Request,
    Session,
    Trace,
    TraceError,
    TraceHeader,
    iter_records,
    validate_trace,
)

HASH_SCHEME = "weka-local-v1"


def block_hash(trace_id: str, block_id: int) -> str:
    """Hash of local block ``block_id`` of trace ``trace_id`` (16 lowercase hex digits)."""
    data = f"{HASH_SCHEME}\0{trace_id}\0{block_id}".encode()
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def tail_hash(trace_id: str, req_id: str) -> str:
    """Hash of a request's partial last block, unique to that request."""
    data = f"{HASH_SCHEME}\0{trace_id}\0tail\0{req_id}".encode()
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def read_weka(paths: Iterable[str | Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(origin, trace)`` from JSON files, JSONL files, or directories of them.

    Directories are read in sorted file-name order; a JSONL file holds one trace per line.
    ``origin`` names the file (and line) for error messages and session records.
    """
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix in (".json", ".jsonl"))
        else:
            files = [p]
        for f in files:
            if f.suffix == ".jsonl":
                with open(f, encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if line.strip():
                            yield f"{f.name}:{i}", json.loads(line)
            else:
                with open(f, encoding="utf-8") as fh:
                    yield f.name, json.load(fh)


@dataclass
class _Raw:
    agent: str
    index: int
    t: float
    rec: dict[str, Any]
    spawn: tuple[str, float] | None
    """For the first request of a subagent: (spawning agent, spawn time)."""


def _flatten(trace: dict[str, Any], tid: str) -> list[_Raw]:
    out: list[_Raw] = []
    seen_agents: set[str] = set()

    def walk(reqs: list[dict[str, Any]], agent: str, t0: float, spawn: tuple[str, float] | None):
        if agent in seen_agents:
            raise TraceError(f"trace {tid}: duplicate agent id {agent!r}")
        seen_agents.add(agent)
        index = 0
        for q in reqs:
            t = t0 + float(q["t"])
            if q.get("type") == "subagent":
                child = f"{agent}/{q.get('agent_id', f'sub{len(seen_agents)}')}"
                walk(q.get("requests", []), child, t, (agent, t))
                continue
            out.append(_Raw(agent, index, t, q, spawn if index == 0 else None))
            index += 1

    walk(trace.get("requests", []), "main", 0.0, None)
    return out


def _longest_shared(ids: list[int], index: dict[tuple[int, int], int]) -> int:
    """Longest prefix of ``ids``, in blocks, shared with any request already in ``index``.

    ``index`` maps (position, block id) to the latest request holding that block there.
    Block IDs behave like chained hashes (see docs/PUBLIC_TRACES.md), so sharing block
    ``k`` implies sharing every block before it and a binary search suffices.
    """
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if (mid - 1, ids[mid - 1]) in index:
            lo = mid
        else:
            hi = mid - 1
    return lo


def convert_trace(
    trace: dict[str, Any],
    origin: str,
    source: str,
    license_: str,
    raw_out: dict[str, dict[str, Any]] | None = None,
) -> tuple[Session, list[Request], int]:
    """Convert one WEKA-format trace. Returns the session, its requests, and prefix tokens.

    If ``raw_out`` is given, it receives each request's source record by ``req_id``, for
    analyses that need fields the trace format does not carry (``stop``, ``api_time``).
    """
    tid = str(trace.get("id") or Path(origin).stem)
    if trace.get("block_size", BLOCK_TOKENS) != BLOCK_TOKENS:
        raise TraceError(f"{origin}: block_size {trace.get('block_size')} is not {BLOCK_TOKENS}")
    raws = _flatten(trace, tid)
    # Arrival order; ties keep the spawning agent's requests first, then file order.
    raws.sort(key=lambda r: (r.t, r.agent.count("/"), r.agent, r.index))

    def rid(r: _Raw) -> str:
        agent = r.agent.replace("main", "m", 1).replace("/", "-")
        return f"{tid}-{agent}-{r.index:04d}"

    by_agent: dict[str, list[tuple[_Raw, Request]]] = {}
    positions: dict[str, dict[tuple[int, int], int]] = {}
    built: list[tuple[_Raw, Request]] = []
    for r in raws:
        q = r.rec
        n_in, ids = int(q["in"]), [int(x) for x in q.get("hash_ids", [])]
        if n_in < 1:
            raise TraceError(f"{origin}: request {rid(r)} has {n_in} input tokens")
        if not (BLOCK_TOKENS * len(ids) <= n_in < BLOCK_TOKENS * (len(ids) + 1)):
            raise TraceError(
                f"{origin}: request {rid(r)}: {n_in} input tokens but {len(ids)} full blocks"
            )
        blocks = [block_hash(tid, i) for i in ids]
        if n_in % BLOCK_TOKENS:
            blocks.append(tail_hash(tid, rid(r)))

        parent: Request | None = None
        pattern: str | None = None
        index = positions.setdefault(r.agent, {})
        best = _longest_shared(ids, index)
        if best:
            p_raw, parent = by_agent[r.agent][index[(best - 1, ids[best - 1])]]
            p_full = len(p_raw.rec.get("hash_ids", []))
            if best == p_full and n_in > parent.prompt_tokens:
                pattern = "append"
            elif best == p_full == len(ids) and n_in == parent.prompt_tokens:
                pattern = "retry"
            else:
                pattern = "rewrite"
        elif r.spawn is not None:
            spawner, t_spawn = r.spawn
            cands = [preq for pr, preq in by_agent.get(spawner, []) if pr.t <= t_spawn]
            if cands:
                parent, pattern = cands[-1], "fork"

        req = Request(
            req_id=rid(r),
            session_id=tid,
            parent_req_id=parent.req_id if parent else None,
            branch_id=0,
            t_arrival_s=round(r.t, 6),
            prompt_tokens=n_in,
            blocks=tuple(blocks),
            segments=None,
            output_tokens=int(q.get("out", 0)),
            tool_wait_s=None,
            pattern=pattern,  # type: ignore[arg-type]
        )
        agent_reqs = by_agent.setdefault(r.agent, [])
        for k, b in enumerate(ids):
            index[(k, b)] = len(agent_reqs)
        agent_reqs.append((r, req))
        built.append((r, req))
        if raw_out is not None:
            raw_out[req.req_id] = q

    # Branch ids: 0 for the main agent, then one per subagent in order of first arrival.
    branch: dict[str, int] = {"main": 0}
    for r, _ in built:
        branch.setdefault(r.agent, len(branch))
    # tool_wait_s from the earliest non-fork child's think_time.
    wait: dict[str, float] = {}
    for r, req in built:
        if req.parent_req_id and req.pattern != "fork" and req.parent_req_id not in wait:
            tt = r.rec.get("think_time")
            if tt is not None:
                wait[req.parent_req_id] = round(max(float(tt), 0.0), 6)
    requests = [
        replace(req, branch_id=branch[r.agent], tool_wait_s=wait.get(req.req_id))
        for r, req in built
    ]
    models = trace.get("models") or sorted({str(r.rec.get("model", "")) for r in raws} - {""})
    session = Session(
        session_id=tid,
        t_start_s=0.0,
        model_family=",".join(models),
        source=f"{source}:{origin}",
        license=license_,
    )
    prefix = int(trace.get("tool_tokens", 0)) + int(trace.get("system_tokens", 0))
    return session, requests, prefix


def iter_import(
    paths: Iterable[str | Path],
    *,
    source: str,
    license_: str,
    session_interval_s: float = 0.0,
    limit: int | None = None,
) -> Iterator[tuple[str, Session, list[Request], int]]:
    """Convert traces one at a time: yields ``(origin, session, requests, prefix tokens)``.

    Session ``i`` starts at ``i * session_interval_s``, an assumption: the source records
    each trace relative to its own start.
    """
    if session_interval_s < 0:
        raise ValueError("session_interval_s must be >= 0")
    seen: set[str] = set()
    for i, (origin, raw) in enumerate(read_weka(paths)):
        if limit is not None and i >= limit:
            break
        session, reqs, n_prefix = convert_trace(raw, origin, source, license_)
        if session.session_id in seen:
            raise TraceError(f"{origin}: duplicate trace id {session.session_id}")
        seen.add(session.session_id)
        yield origin, replace(session, t_start_s=round(i * session_interval_s, 6)), reqs, n_prefix


def _header(
    generator: str,
    source: str,
    license_: str,
    session_interval_s: float,
    inputs: list[str],
    prefix: dict[str, int],
) -> TraceHeader:
    return TraceHeader(
        hash_scheme=HASH_SCHEME,
        generator=generator,
        params={
            "source": source,
            "license": license_,
            "inputs": inputs,
            "session_interval_s": session_interval_s,
            "prefix_tokens": prefix,
            "provenance": {
                "recorded": "block structure, token counts, per-trace timing",
                "derived": "parent_req_id, pattern, tool_wait_s",
                "assumption": "session start times",
            },
        },
    )


def import_weka(
    paths: Iterable[str | Path],
    *,
    source: str,
    license_: str,
    session_interval_s: float = 0.0,
    limit: int | None = None,
    generator: str = "state-bench import weka",
) -> Trace:
    """Import WEKA-format traces into one in-memory trace. See :func:`write_import`."""
    sessions: list[Session] = []
    requests: list[Request] = []
    prefix: dict[str, int] = {}
    inputs: list[str] = []
    for origin, session, reqs, n_prefix in iter_import(
        paths,
        source=source,
        license_=license_,
        session_interval_s=session_interval_s,
        limit=limit,
    ):
        sessions.append(session)
        requests.extend(reqs)
        prefix[session.session_id] = n_prefix
        inputs.append(origin)
    header = _header(generator, source, license_, session_interval_s, inputs, prefix)
    return Trace(header=header, sessions=sessions, requests=requests)


def write_import(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    source: str,
    license_: str,
    session_interval_s: float = 0.0,
    limit: int | None = None,
    generator: str = "state-bench import weka",
) -> tuple[int, int]:
    """Import WEKA-format traces into a JSONL file, one trace in memory at a time.

    Each session is validated on its own before it is written. Returns the numbers of
    sessions and requests. The output is byte-for-byte deterministic and matches
    ``dump_trace(import_weka(...))``.
    """
    out = Path(output)
    body = out.with_name(out.name + ".requests.tmp")
    sessions: list[Session] = []
    prefix: dict[str, int] = {}
    inputs: list[str] = []
    n_req = 0
    try:
        with open(body, "w", encoding="utf-8") as fb:
            for origin, session, reqs, n_prefix in iter_import(
                paths,
                source=source,
                license_=license_,
                session_interval_s=session_interval_s,
                limit=limit,
            ):
                part = Trace(TraceHeader(hash_scheme=HASH_SCHEME), [session], reqs)
                errs = validate_trace(part)
                if errs:
                    raise TraceError(f"{origin}: " + "; ".join(errs[:5]))
                for rec in iter_records(part):
                    if rec["type"] == "request":
                        fb.write(_line(rec))
                sessions.append(session)
                prefix[session.session_id] = n_prefix
                inputs.append(origin)
                n_req += len(reqs)
        header = _header(generator, source, license_, session_interval_s, inputs, prefix)
        with open(out, "w", encoding="utf-8") as fo, open(body, encoding="utf-8") as fb:
            for rec in iter_records(Trace(header, sessions, [])):
                fo.write(_line(rec))
            shutil.copyfileobj(fb, fo)
    finally:
        body.unlink(missing_ok=True)
    return len(sessions), n_req


def _line(rec: dict[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"

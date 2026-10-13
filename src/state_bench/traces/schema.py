"""Trace format, version 0.1.

A trace records the structure and timing of LLM requests, never their content.
Prefix sharing is expressed with chained hashes over 64-token blocks, so traces
recorded from real sessions can be published without exposing prompts.

A trace file is JSONL. Each line is one record, distinguished by ``type``:

* ``header`` (first line only): format version, hash scheme, generator and its
  parameters.
* ``session``: per-session metadata, including source and license.
* ``request``: one inference request. Within a session, requests appear in
  non-decreasing ``t_arrival_s`` order and a parent always precedes its children.

This module provides the dataclasses, a JSON Schema for per-line structural
validation (:data:`JSON_SCHEMA`), cross-record semantic validation
(:func:`validate_trace`), and readers and writers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "0.1"
BLOCK_TOKENS = 64

SegmentKind = Literal["system", "user", "assistant", "tool_result"]
Pattern = Literal["append", "fork", "retry", "rewrite"]

SEGMENT_KINDS: tuple[str, ...] = ("system", "user", "assistant", "tool_result")
PATTERNS: tuple[str, ...] = ("append", "fork", "retry", "rewrite")

_HASH_RE = "^[0-9a-f]{16}$"


@dataclass(frozen=True)
class Segment:
    """One message in a request's input, covering tokens ``[start_tok, end_tok)``."""

    kind: SegmentKind
    start_tok: int
    end_tok: int

    @property
    def length(self) -> int:
        return self.end_tok - self.start_tok


@dataclass(frozen=True)
class Session:
    session_id: str
    t_start_s: float
    """Session start, relative to the start of the trace."""
    model_family: str
    source: str
    license: str


@dataclass(frozen=True)
class Request:
    req_id: str
    session_id: str
    parent_req_id: str | None
    branch_id: int
    t_arrival_s: float
    """Arrival time relative to the start of the session."""
    prompt_tokens: int
    blocks: tuple[str, ...]
    """Chained hashes of the input in 64-token blocks (16 lowercase hex digits).

    The last block may be partial; its length is derived from ``prompt_tokens``.
    """
    segments: tuple[Segment, ...] | None
    """Message boundaries, or ``None`` when the source does not record them."""
    output_tokens: int
    tool_wait_s: float | None
    """Gap between the end of this request's output and the next request's arrival.

    ``None`` for the last request of a chain or when the source does not record it.
    """
    pattern: Pattern | None
    """Relationship to the parent request; ``None`` for a root request."""

    @property
    def last_block_tokens(self) -> int:
        """Number of tokens in the last block (1 to 64)."""
        return self.prompt_tokens - BLOCK_TOKENS * (len(self.blocks) - 1)

    @property
    def full_blocks(self) -> tuple[str, ...]:
        """Blocks that hold exactly 64 tokens. Only full blocks can match across requests."""
        if self.last_block_tokens == BLOCK_TOKENS:
            return self.blocks
        return self.blocks[:-1]


@dataclass
class TraceHeader:
    schema_version: str = SCHEMA_VERSION
    block_tokens: int = BLOCK_TOKENS
    hash_scheme: str = "blake2b64-chain-v1"
    """Block hashes from different schemes are not comparable."""
    generator: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    """Generator parameters. Synthetic traces label them ``provenance: assumption``."""


@dataclass
class Trace:
    header: TraceHeader
    sessions: list[Session]
    requests: list[Request]

    def global_arrivals(self) -> list[tuple[float, Request]]:
        """All requests sorted by absolute arrival time (session start + relative arrival)."""
        start = {s.session_id: s.t_start_s for s in self.sessions}
        out = [(start[r.session_id] + r.t_arrival_s, r) for r in self.requests]
        out.sort(key=lambda x: x[0])
        return out

    def requests_by_session(self) -> dict[str, list[Request]]:
        out: dict[str, list[Request]] = {s.session_id: [] for s in self.sessions}
        for r in self.requests:
            out.setdefault(r.session_id, []).append(r)
        return out


# ---------------------------------------------------------------------------
# JSON Schema (per-line structure)
# ---------------------------------------------------------------------------

_HEADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["type", "schema_version", "block_tokens", "hash_scheme", "generator", "params"],
    "additionalProperties": False,
    "properties": {
        "type": {"const": "header"},
        "schema_version": {"type": "string", "pattern": r"^\d+\.\d+$"},
        "block_tokens": {"const": BLOCK_TOKENS},
        "hash_scheme": {"type": "string", "minLength": 1},
        "generator": {"type": "string"},
        "params": {"type": "object"},
    },
}

_SESSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["type", "session_id", "t_start_s", "model_family", "source", "license"],
    "additionalProperties": False,
    "properties": {
        "type": {"const": "session"},
        "session_id": {"type": "string", "minLength": 1},
        "t_start_s": {"type": "number", "minimum": 0},
        "model_family": {"type": "string"},
        "source": {"type": "string", "minLength": 1},
        "license": {"type": "string", "minLength": 1},
    },
}

_SEGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kind", "start_tok", "end_tok"],
    "additionalProperties": False,
    "properties": {
        "kind": {"enum": list(SEGMENT_KINDS)},
        "start_tok": {"type": "integer", "minimum": 0},
        "end_tok": {"type": "integer", "minimum": 1},
    },
}

_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "type",
        "req_id",
        "session_id",
        "parent_req_id",
        "branch_id",
        "t_arrival_s",
        "prompt_tokens",
        "blocks",
        "segments",
        "output_tokens",
        "tool_wait_s",
        "pattern",
    ],
    "additionalProperties": False,
    "properties": {
        "type": {"const": "request"},
        "req_id": {"type": "string", "minLength": 1},
        "session_id": {"type": "string", "minLength": 1},
        "parent_req_id": {"type": ["string", "null"]},
        "branch_id": {"type": "integer", "minimum": 0},
        "t_arrival_s": {"type": "number", "minimum": 0},
        "prompt_tokens": {"type": "integer", "minimum": 1},
        "blocks": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "pattern": _HASH_RE},
        },
        "segments": {"type": ["array", "null"], "items": _SEGMENT_SCHEMA},
        "output_tokens": {"type": "integer", "minimum": 0},
        "tool_wait_s": {"type": ["number", "null"], "minimum": 0},
        "pattern": {"enum": [*PATTERNS, None]},
    },
}

JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"https://github.com/outerarc-ai/state-bench/schema/trace-{SCHEMA_VERSION}.json",
    "title": "state-bench trace record",
    "description": "One line of a state-bench JSONL trace file.",
    "oneOf": [_HEADER_SCHEMA, _SESSION_SCHEMA, _REQUEST_SCHEMA],
}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _request_to_dict(r: Request) -> dict[str, Any]:
    d = asdict(r)
    d["blocks"] = list(r.blocks)
    d["segments"] = None if r.segments is None else [asdict(s) for s in r.segments]
    return {"type": "request", **d}


def _request_from_dict(d: dict[str, Any]) -> Request:
    segs = d["segments"]
    return Request(
        req_id=d["req_id"],
        session_id=d["session_id"],
        parent_req_id=d["parent_req_id"],
        branch_id=d["branch_id"],
        t_arrival_s=float(d["t_arrival_s"]),
        prompt_tokens=d["prompt_tokens"],
        blocks=tuple(d["blocks"]),
        segments=None if segs is None else tuple(Segment(**s) for s in segs),
        output_tokens=d["output_tokens"],
        tool_wait_s=None if d["tool_wait_s"] is None else float(d["tool_wait_s"]),
        pattern=d["pattern"],
    )


def iter_records(trace: Trace) -> Iterator[dict[str, Any]]:
    yield {"type": "header", **asdict(trace.header)}
    for s in trace.sessions:
        yield {"type": "session", **asdict(s)}
    for r in trace.requests:
        yield _request_to_dict(r)


def dump_trace(trace: Trace, path: str | Path) -> None:
    """Write a trace as JSONL. The output is byte-for-byte deterministic."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in iter_records(trace):
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def parse_records(records: Iterable[dict[str, Any]]) -> Trace:
    header: TraceHeader | None = None
    sessions: list[Session] = []
    requests: list[Request] = []
    for i, rec in enumerate(records):
        kind = rec.get("type")
        if i == 0:
            if kind != "header":
                raise TraceError("line 1: the first record must be a header")
            header = TraceHeader(**{k: v for k, v in rec.items() if k != "type"})
            continue
        if kind == "session":
            sessions.append(Session(**{k: v for k, v in rec.items() if k != "type"}))
        elif kind == "request":
            requests.append(_request_from_dict(rec))
        else:
            raise TraceError(f"line {i + 1}: unexpected record type {kind!r}")
    if header is None:
        raise TraceError("empty trace")
    return Trace(header=header, sessions=sessions, requests=requests)


def load_trace(path: str | Path, *, check: bool = True) -> Trace:
    """Read a JSONL trace. With ``check=True``, run structural and semantic validation."""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if check:
        check_json_schema(records)
    trace = parse_records(records)
    if check:
        errs = validate_trace(trace)
        if errs:
            raise TraceError("semantic validation failed:\n" + "\n".join(errs[:20]))
    return trace


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TraceError(ValueError):
    """Raised when a trace is malformed."""


def check_json_schema(records: Iterable[dict[str, Any]]) -> None:
    """Validate each record against :data:`JSON_SCHEMA`; report the first failing line."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(JSON_SCHEMA)
    for i, rec in enumerate(records):
        errs = sorted(validator.iter_errors(rec), key=lambda e: e.path)
        if errs:
            raise TraceError(f"line {i + 1}: invalid record: {errs[0].message}")


def validate_trace(trace: Trace) -> list[str]:
    """Cross-record validation. Returns a list of problems; empty means valid.

    Checks:

    1. Schema major version and block size.
    2. Per request: ``len(blocks) == ceil(prompt_tokens / 64)``; segments are
       contiguous and cover the whole input.
    3. Sessions are declared; ``req_id`` is unique; a parent exists, belongs to the
       same session, appears earlier, and does not arrive later than its child.
    4. ``pattern`` is set exactly when ``parent_req_id`` is set.
    5. Prefix relations: ``append`` contains every full block of its parent;
       ``retry`` has the same blocks as its parent. ``fork`` and ``rewrite`` may
       share less than one full block, so they are not checked at block level.
    """
    errs: list[str] = []
    h = trace.header
    if h.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        errs.append(f"unsupported schema_version {h.schema_version} (reader: {SCHEMA_VERSION})")
    if h.block_tokens != BLOCK_TOKENS:
        errs.append(f"block_tokens must be {BLOCK_TOKENS}")

    session_ids = {s.session_id for s in trace.sessions}
    if len(session_ids) != len(trace.sessions):
        errs.append("duplicate session_id")

    seen: dict[str, Request] = {}
    for r in trace.requests:
        where = f"request {r.req_id}"
        if r.req_id in seen:
            errs.append(f"{where}: duplicate req_id")
            continue
        if r.session_id not in session_ids:
            errs.append(f"{where}: session {r.session_id} is not declared")

        n_expect = math.ceil(r.prompt_tokens / BLOCK_TOKENS)
        if len(r.blocks) != n_expect:
            errs.append(
                f"{where}: {r.prompt_tokens} tokens need {n_expect} blocks, got {len(r.blocks)}"
            )

        if r.segments is not None:
            errs.extend(f"{where}: {e}" for e in _check_segments(r.segments, r.prompt_tokens))

        if (r.parent_req_id is None) != (r.pattern is None):
            errs.append(f"{where}: parent_req_id and pattern must both be set or both be null")

        if r.parent_req_id is not None:
            p = seen.get(r.parent_req_id)
            if p is None:
                errs.append(f"{where}: parent {r.parent_req_id} is missing or appears later")
            else:
                if p.session_id != r.session_id:
                    errs.append(f"{where}: parent belongs to another session")
                if p.t_arrival_s > r.t_arrival_s:
                    errs.append(f"{where}: arrives earlier than its parent")
                if r.pattern is not None:
                    errs.extend(f"{where}: {e}" for e in _check_pattern(r, p))
        seen[r.req_id] = r

    for sid, reqs in trace.requests_by_session().items():
        for a, b in pairwise(reqs):
            if b.t_arrival_s < a.t_arrival_s:
                errs.append(
                    f"session {sid}: request {b.req_id} arrives earlier than the one before it"
                )
    return errs


def _check_segments(segs: tuple[Segment, ...], prompt_tokens: int) -> list[str]:
    if not segs:
        return ["segments is an empty list; use null when boundaries are unknown"]
    errs = []
    if segs[0].start_tok != 0:
        errs.append("the first segment does not start at token 0")
    for a, b in pairwise(segs):
        if b.start_tok != a.end_tok:
            errs.append(f"segments are not contiguous: {a.end_tok} -> {b.start_tok}")
    for s in segs:
        if s.end_tok <= s.start_tok:
            errs.append(f"empty or negative segment [{s.start_tok}, {s.end_tok})")
    if segs[-1].end_tok != prompt_tokens:
        errs.append(f"segments end at {segs[-1].end_tok} but prompt_tokens={prompt_tokens}")
    return errs


def shared_full_blocks(a: Request, b: Request) -> int:
    """Length, in blocks, of the shared full-block prefix of two requests.

    Hashes are chained, so the first mismatch ends the shared prefix.
    """
    n = 0
    for x, y in zip(a.full_blocks, b.full_blocks, strict=False):
        if x != y:
            break
        n += 1
    return n


def _check_pattern(r: Request, p: Request) -> list[str]:
    shared = shared_full_blocks(r, p)
    if r.pattern == "append":
        need = len(p.full_blocks)
        if shared < need or r.prompt_tokens <= p.prompt_tokens:
            return [f"append must extend all {need} full blocks of its parent; shares {shared}"]
    elif r.pattern == "retry":
        if r.blocks != p.blocks:
            return ["retry must have the same input blocks as its parent"]
    return []

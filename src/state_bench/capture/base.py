"""Engine capture: sources observe a serving system, a sink builds a trace.

A capture source observes requests against a live system and emits
:class:`CapturedRequest` values (token-level prompt representation plus the
measured times). The :class:`TraceSink` turns them into a state-bench trace:
it assigns ids, infers the parent and pattern from chained block hashes,
fills ``tool_wait_s``, writes the JSONL trace plus a sidecar of measured
annotations, and validates before writing.

Provenance: captured times and lengths are *measured*. Lineage and pattern
are *derived* from the observed prefixes by the documented heuristic below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..traces.schema import (
    BLOCK_TOKENS,
    Request,
    Segment,
    Session,
    Trace,
    TraceError,
    TraceHeader,
    dump_trace,
    validate_trace,
)

GENERATOR_VERSION = "capture-0.1"
HASH_SCHEME = "tokenids-blake2b64-chain-v1"
PROVENANCE = "measured"

#: Pattern inference for a request sharing a proper prefix of its candidate
#: parent's full blocks: shorter than the parent implies a compaction
#: (rewrite), otherwise a branch (fork). Recorded in the header params.
FORK_REWRITE_RULE = "shared-full-prefix, shorter than parent -> rewrite, else fork"

_ZERO = b"\x00" * 8


def shared_full_block_hashes(a: Sequence[str], b: Sequence[str]) -> int:
    """Length, in blocks, of the shared prefix of two chained block-hash tuples."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _block_hash(prev: bytes, n_tokens: int, ids: Sequence[int]) -> bytes:
    h = hashlib.blake2b(digest_size=8)
    h.update(prev)
    h.update(struct.pack("<H", n_tokens))
    for t in ids:
        h.update(struct.pack("<I", t))
    return h.digest()


class TokenBlockHasher:
    """Chained 64-token block hashing over a token-id stream.

    The block hash is ``blake2b(prev || u16(block length) || token ids as
    little-endian u32)`` with an 8-byte digest, so a partial tail block is
    distinguishable from a full block at the same position. The scheme
    (see :data:`HASH_SCHEME`) hashes token ids, not raw bytes, and is not
    comparable with the ``blake2b64-chain-v1`` used by synthetic traces.
    """

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._full: list[bytes] = []

    @property
    def n_tokens(self) -> int:
        return len(self._ids)

    def extend(self, ids: Sequence[int]) -> None:
        self._ids.extend(ids)
        while (len(self._full) + 1) * BLOCK_TOKENS <= len(self._ids):
            i = len(self._full)
            prev = self._full[-1] if self._full else _ZERO
            self._full.append(
                _block_hash(
                    prev, BLOCK_TOKENS, self._ids[i * BLOCK_TOKENS : (i + 1) * BLOCK_TOKENS]
                )
            )

    def blocks(self) -> tuple[str, ...]:
        """Block hashes of the stream so far, including a partial tail."""
        out = [b.hex() for b in self._full]
        tail = self._ids[len(self._full) * BLOCK_TOKENS :]
        if tail:
            prev = self._full[-1] if self._full else _ZERO
            out.append(_block_hash(prev, len(tail), tail).hex())
        return tuple(out)


@dataclass(frozen=True)
class CapturedRequest:
    """One observed request. Times are relative to the session start."""

    session_id: str
    t_arrival_s: float
    t_done_s: float
    prompt_tokens: int
    blocks: tuple[str, ...]
    segments: tuple[Segment, ...] | None
    output_tokens: int
    """Completion tokens of the response; 0 when the source does not report them."""
    annotations: dict[str, object] = field(default_factory=dict)
    """Measured engine-reported extras (e.g. the raw ``usage`` object).
    Written to the sidecar file, never into the trace records."""


class TraceSink:
    """Accumulates captured requests into a trace, inferring lineage by prefix.

    Requests must be submitted in non-decreasing arrival order per session.
    The parent of a new request is the earlier request of the same session
    with the longest shared full-block prefix (ties: the most recent one).
    Pattern inference:

    * no shared full block  -> root (no parent);
    * identical blocks     -> ``retry``;
    * all of the parent's full blocks shared and the prompt is longer
                            -> ``append``;
    * a proper prefix shared -> ``rewrite`` when the prompt is shorter than
      the parent's, else ``fork`` (:data:`FORK_REWRITE_RULE`).

    ``branch_id``: roots are 0; each ``fork`` takes the session's next
    branch number; ``append``/``retry``/``rewrite`` keep the parent's.
    ``tool_wait_s`` is set on a parent when its first ``append``/``retry``
    child arrives: child arrival minus parent completion.
    """

    def __init__(
        self,
        *,
        model_family: str,
        source: str,
        license_: str,
        hash_scheme: str = HASH_SCHEME,
        params: dict[str, object] | None = None,
    ) -> None:
        self.model_family = model_family
        self.source = source
        self.license = license_
        self.hash_scheme = hash_scheme
        self.params = {
            "provenance": PROVENANCE,
            "lineage": "longest shared full-block prefix (ties: most recent)",
            "fork_rewrite_rule": FORK_REWRITE_RULE,
            **(params or {}),
        }
        self._sessions: dict[str, Session] = {}
        self._order: list[Request] = []
        self._by_session: dict[str, list[Request]] = {}
        self._last_arrival: dict[str, float] = {}
        self._done: dict[str, float] = {}
        self._waits: dict[str, float] = {}
        self._seq: dict[str, int] = {}
        self._branch_next: dict[str, int] = {}
        self._annotations: list[tuple[str, dict[str, object]]] = []

    # -- submission ---------------------------------------------------------

    def start_session(self, session_id: str, t_start_s: float) -> None:
        if session_id in self._sessions:
            raise TraceError(f"duplicate session_id {session_id!r}")
        self._sessions[session_id] = Session(
            session_id=session_id,
            t_start_s=round(t_start_s, 6),
            model_family=self.model_family,
            source=self.source,
            license=self.license,
        )
        self._by_session[session_id] = []
        self._seq[session_id] = 0
        self._branch_next[session_id] = 1

    def request(self, req: CapturedRequest) -> Request:
        if req.session_id not in self._sessions:
            raise TraceError(f"request {req.session_id}: session is not declared")
        if req.t_arrival_s < 0 or req.t_done_s < req.t_arrival_s:
            raise TraceError(
                f"request {req.session_id}: invalid times "
                f"(arrival {req.t_arrival_s}, done {req.t_done_s})"
            )
        last = self._last_arrival.get(req.session_id, 0.0)
        if req.t_arrival_s < last:
            raise TraceError(
                f"request {req.session_id}: arrives before the previous request "
                f"({req.t_arrival_s} < {last})"
            )
        self._last_arrival[req.session_id] = req.t_arrival_s

        parent, shared = self._find_parent(req)
        pattern = self._pattern(req, parent, shared)
        if parent is None or pattern is None:
            branch = 0
        elif pattern == "fork":
            branch = self._branch_next[req.session_id]
            self._branch_next[req.session_id] += 1
        else:
            branch = parent.branch_id

        r = Request(
            req_id=f"{req.session_id}-r{self._seq[req.session_id]:04d}",
            session_id=req.session_id,
            parent_req_id=parent.req_id if parent else None,
            branch_id=branch,
            t_arrival_s=round(req.t_arrival_s, 6),
            prompt_tokens=req.prompt_tokens,
            blocks=req.blocks,
            segments=req.segments,
            output_tokens=req.output_tokens,
            tool_wait_s=None,
            pattern=pattern,
        )
        if (
            parent is not None
            and pattern in ("append", "retry")
            and parent.req_id not in self._waits
        ):
            self._waits[parent.req_id] = round(req.t_arrival_s - self._done[parent.req_id], 6)
        self._order.append(r)
        self._by_session[req.session_id].append(r)
        self._done[r.req_id] = req.t_done_s
        self._seq[req.session_id] += 1
        if req.annotations:
            self._annotations.append((r.req_id, req.annotations))
        return r

    # -- lineage ------------------------------------------------------------

    def _find_parent(self, req: CapturedRequest) -> tuple[Request | None, int]:
        best: Request | None = None
        best_shared = 0
        for p in self._by_session[req.session_id]:
            if p.t_arrival_s > req.t_arrival_s:
                continue
            shared = shared_full_block_hashes(p.blocks, req.blocks)
            # Arrival-ordered scan: a later tie replaces the earlier one, so
            # ties resolve to the most recent parent.
            if shared > best_shared or (shared == best_shared and shared > 0):
                best, best_shared = p, shared
        if best is None or best_shared == 0:
            return None, 0
        return best, best_shared

    @staticmethod
    def _pattern(req: CapturedRequest, parent: Request | None, shared: int) -> str | None:
        if parent is None:
            return None
        if req.blocks == parent.blocks:
            return "retry"
        if shared == len(parent.full_blocks) and req.prompt_tokens > parent.prompt_tokens:
            return "append"
        if req.prompt_tokens < parent.prompt_tokens:
            return "rewrite"
        return "fork"

    # -- output ---------------------------------------------------------------

    def finish(self, path: str | Path) -> Trace:
        """Validate and write the trace (plus the sidecar if any annotations)."""
        requests = [
            dataclasses.replace(r, tool_wait_s=self._waits[r.req_id])
            if r.req_id in self._waits
            else r
            for r in self._order
        ]
        trace = Trace(
            header=TraceHeader(
                hash_scheme=self.hash_scheme,
                generator=GENERATOR_VERSION,
                params=self.params,
            ),
            sessions=list(self._sessions.values()),
            requests=requests,
        )
        errs = validate_trace(trace)
        if errs:
            raise TraceError("captured trace failed validation:\n" + "\n".join(errs[:20]))
        dump_trace(trace, path)
        if self._annotations:
            sidecar = Path(str(path) + ".annotations.json")
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(dict(self._annotations), f, ensure_ascii=False, indent=1)
                f.write("\n")
        return trace

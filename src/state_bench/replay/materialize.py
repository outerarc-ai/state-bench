"""Replay materialization: a content-free trace into a structure-faithful token stream.

A trace records chained block hashes, never tokens (see ``docs/TRACE_FORMAT.md``).
An engine, though, takes token ids, so replaying a trace needs tokens that reproduce
the trace's prefix reuse. This module assigns every 64-token block hash a deterministic
token sequence. Block hashes are chained, so requests sharing leading blocks share those
blocks' hashes and therefore the same tokens: an engine's prefix cache sees the reuse the
trace records, and (up to hash collisions) no reuse it does not.

Two reuse relations are reproduced:

* **Shared blocks.** Any two requests with the same leading block hashes get the same
  leading tokens, through a memo keyed by the block-hash prefix.
* **``append``/``retry`` lineage.** The trace guarantees such a request starts with its
  parent's whole prompt, tail included. The child's tokens are copied from the parent and
  continued from the parent's end, so even a parent that ends mid-block is shared exactly
  -- as the simulator credits it (``sim/core.py``).

Token *identity* is an arbitrary but deterministic placeholder (provenance: assumption).
Prefix sharing, prompt lengths, and arrival timing are copied from the trace, so they
inherit the source trace's provenance. Every id lies in ``[0, vocab_size)`` so the
engine's embedding lookup accepts it.

Output is the SGLang Simulator "Autobench" timestamped JSONL contract: one row per
request with ``prompt`` (token ids), ``prompt_len``, ``output_len``, and ``timestamp``
in milliseconds. It is consumed by
``benchmark/simulator/bench_serving.py --dataset-name autobench --use-trace-timestamps``
and by the simulator itself.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..traces.schema import BLOCK_TOKENS, Request, Trace

MATERIALIZER_VERSION = "materialize-0.1"
TOKEN_SCHEME = "shake128-blockhash-v1"
"""Block tokens are ``shake128(block_hash || u16(length))`` reduced into the vocabulary."""

LINKS = ("append", "retry")
"""Patterns whose children repeat the parent's whole prompt, tail included."""


@dataclass(frozen=True)
class MaterializedRequest:
    """One request as a token-id prompt, positioned on the trace's global timeline."""

    req_id: str
    session_id: str
    prompt: tuple[int, ...]
    output_len: int
    """Requested completion length. ``0`` means the trace did not record one."""
    timestamp_ms: float
    """Arrival on the global timeline, in milliseconds from the trace's origin."""


@dataclass(frozen=True)
class MaterializedTrace:
    vocab_size: int
    requests: tuple[MaterializedRequest, ...]
    """Requests in non-decreasing global arrival order."""

    def by_req_id(self) -> dict[str, MaterializedRequest]:
        return {m.req_id: m for m in self.requests}


def _block_token_ids(hash_hex: str, n_tokens: int, vocab_size: int) -> tuple[int, ...]:
    """Deterministic token ids for one block: ``n_tokens`` ids from the block hash."""
    seed = hashlib.shake_128(bytes.fromhex(hash_hex) + struct.pack("<H", n_tokens)).digest(
        4 * n_tokens
    )
    return tuple(
        int.from_bytes(seed[i : i + 4], "little") % vocab_size for i in range(0, 4 * n_tokens, 4)
    )


class _Materializer:
    """Builds one token stream per request, sharing tokens along the trace's structure."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.by_id: dict[str, Request] = {}
        self.out: dict[str, tuple[int, ...]] = {}
        self.memo: dict[tuple[str, ...], tuple[int, ...]] = {}
        """Block tokens by block-hash prefix: equal hashes give equal tokens."""

    def _block(self, r: Request, i: int) -> tuple[int, ...]:
        """Tokens of block ``i`` (64, or the tail length), memoized by hash prefix."""
        key = r.blocks[: i + 1]
        cached = self.memo.get(key)
        if cached is None:
            n = BLOCK_TOKENS if i < len(r.blocks) - 1 else r.last_block_tokens
            cached = _block_token_ids(r.blocks[i], n, self.vocab_size)
            self.memo[key] = cached
        return cached

    def tokens_for(self, r: Request) -> tuple[int, ...]:
        parent = self.by_id.get(r.parent_req_id) if r.parent_req_id else None
        ids: list[int] = []
        if parent is not None and r.pattern in LINKS and parent.req_id in self.out:
            ids = list(self.out[parent.req_id])  # the parent's whole prompt, tail included

        i = len(ids) // BLOCK_TOKENS
        if len(ids) % BLOCK_TOKENS:
            # The inherited prefix leaves block ``i`` open. Keep the inherited tokens for
            # it and take the remainder from the block's own hash, then re-memo the
            # spliced block so a later sharer of this hash sees the same tokens.
            open_tokens = ids[i * BLOCK_TOKENS :]
            block = self._block(r, i)
            spliced = (*open_tokens, *block[len(open_tokens) :])
            self.memo[r.blocks[: i + 1]] = spliced
            ids[i * BLOCK_TOKENS :] = spliced
            i += 1
        for j in range(i, len(r.blocks)):
            ids.extend(self._block(r, j))
        return tuple(ids)


def materialize(trace: Trace, *, vocab_size: int) -> MaterializedTrace:
    """Materialize every request, ordered by global arrival (session start + relative)."""
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    m = _Materializer(vocab_size)
    m.by_id = {r.req_id: r for r in trace.requests}
    # ``validate_trace`` guarantees a parent precedes its children, so one pass suffices.
    # A request whose parent is absent (an unchecked trace) is materialized as a root.
    for r in trace.requests:
        m.out[r.req_id] = m.tokens_for(r)
    requests = [
        MaterializedRequest(
            req_id=r.req_id,
            session_id=r.session_id,
            prompt=m.out[r.req_id],
            output_len=r.output_tokens,
            timestamp_ms=round(abs_t * 1000.0, 6),
        )
        for abs_t, r in trace.global_arrivals()
    ]
    return MaterializedTrace(vocab_size=vocab_size, requests=tuple(requests))


# ---------------------------------------------------------------------------
# Prefix fidelity
# ---------------------------------------------------------------------------


def shared_block_tokens(a: Request, b: Request) -> int:
    """Tokens two requests must share from their common leading block hashes.

    Chained hashes make the first difference end the shared region; a shared tail block
    counts too (its hash encodes its length). This is the block-level share, before the
    whole-prompt ``append``/``retry`` lineage that :func:`check_prefix_fidelity` adds.
    """
    total = 0
    for i, (x, y) in enumerate(zip(a.blocks, b.blocks, strict=False)):
        if x != y:
            break
        total += BLOCK_TOKENS if i < len(a.blocks) - 1 else a.last_block_tokens
    return total


def shared_materialized_prefix(a: Iterable[int], b: Iterable[int]) -> int:
    """Length of the leading run of equal token ids."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def lineage(trace: Trace) -> dict[str, frozenset[str]]:
    """Each request's ``append``/``retry`` ancestors, itself included (as the simulator)."""
    idx = {r.req_id: r for r in trace.requests}
    out: dict[str, frozenset[str]] = {}
    for r in trace.requests:
        p = idx.get(r.parent_req_id) if r.parent_req_id else None
        if p is not None and r.pattern in LINKS and p.req_id in out:
            out[r.req_id] = frozenset({r.req_id}) | out[p.req_id]
        else:
            out[r.req_id] = frozenset({r.req_id})
    return out


#: Above this many pairs, :func:`check_prefix_fidelity` checks lineage only.
MAX_ALL_PAIRS = 2_000_000


def check_prefix_fidelity(
    trace: Trace, mat: MaterializedTrace, *, max_all_pairs: int = MAX_ALL_PAIRS
) -> list[str]:
    """Verify the materialized tokens reproduce the trace's prefix structure.

    Per request, the prompt must have ``prompt_tokens`` ids in ``[0, vocab_size)``. For
    each pair (all pairs when they number few enough, else parent/child pairs), the shared
    token prefix must equal the block-level share widened by any whole-prompt
    ``append``/``retry`` relation. Returns problems; empty means faithful.
    """
    errs: list[str] = []
    by_id = mat.by_req_id()
    for r in trace.requests:
        m = by_id.get(r.req_id)
        if m is None:
            errs.append(f"{r.req_id}: not materialized")
            continue
        if len(m.prompt) != r.prompt_tokens:
            errs.append(f"{r.req_id}: {len(m.prompt)} tokens, expected {r.prompt_tokens}")
        elif any(t < 0 or t >= mat.vocab_size for t in m.prompt):
            errs.append(f"{r.req_id}: token id outside [0, {mat.vocab_size})")

    lin = lineage(trace)
    n = len(trace.requests)
    if n * (n - 1) // 2 <= max_all_pairs:
        pairs: Iterable[tuple[Request, Request]] = _all_pairs(trace.requests)
    else:
        pairs = _parent_child_pairs(trace)
    for a, b in pairs:
        ma, mb = by_id.get(a.req_id), by_id.get(b.req_id)
        if ma is None or mb is None:
            continue
        expected = shared_block_tokens(a, b)
        if a.req_id in lin[b.req_id]:
            expected = max(expected, len(ma.prompt))
        if b.req_id in lin[a.req_id]:
            expected = max(expected, len(mb.prompt))
        expected = min(expected, len(ma.prompt), len(mb.prompt))
        got = shared_materialized_prefix(ma.prompt, mb.prompt)
        if got != expected:
            errs.append(f"{a.req_id}/{b.req_id}: share {got}, expected {expected}")
    return errs


def _all_pairs(reqs: Sequence[Request]) -> Iterable[tuple[Request, Request]]:
    for i, a in enumerate(reqs):
        for b in reqs[i + 1 :]:
            yield a, b


def _parent_child_pairs(trace: Trace) -> Iterable[tuple[Request, Request]]:
    idx = {r.req_id: r for r in trace.requests}
    for r in trace.requests:
        p = idx.get(r.parent_req_id) if r.parent_req_id else None
        if p is not None:
            yield p, r


# ---------------------------------------------------------------------------
# Autobench interchange
# ---------------------------------------------------------------------------


def dump_autobench(mat: MaterializedTrace, path: str | Path) -> None:
    """Write the Autobench timestamped JSONL: one row per request in arrival order.

    ``output_len`` is omitted when zero: the SGLang Simulator loader falls back to its
    own default, so a recorded ``0`` must not be written as an explicit field.
    """
    lines = []
    for m in mat.requests:
        row: dict[str, object] = {"prompt": list(m.prompt), "prompt_len": len(m.prompt)}
        if m.output_len:
            row["output_len"] = m.output_len
        row["timestamp"] = m.timestamp_ms
        lines.append(json.dumps(row))
    Path(path).write_text("\n".join(lines) + "\n")


def write_replay(trace: Trace, path: str | Path, *, vocab_size: int) -> MaterializedTrace:
    """Materialize ``trace``; write the Autobench JSONL and a ``.meta.json`` sidecar."""
    mat = materialize(trace, vocab_size=vocab_size)
    p = Path(path)
    dump_autobench(mat, p)
    meta = {
        "version": MATERIALIZER_VERSION,
        "token_scheme": TOKEN_SCHEME,
        "vocab_size": mat.vocab_size,
        "provenance": {
            "token_ids": "assumption",
            "prompt_structure_and_timing": "inherits the source trace",
        },
        "source_trace": {
            "generator": trace.header.generator,
            "hash_scheme": trace.header.hash_scheme,
            "sessions": len(trace.sessions),
            "requests": len(trace.requests),
        },
        "materialized": {
            "requests": len(mat.requests),
            "tokens": sum(len(m.prompt) for m in mat.requests),
        },
    }
    p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return mat

"""Trace-driven simulator for recurrent-state checkpoint policies.

The simulator replays a trace request by request and tracks two things:

* **KV coverage**: which prefixes have attention KV cached. Version 0 assumes KV
  capacity is unlimited, so coverage only grows.
* **State snapshots**: recurrent-state checkpoints at token positions, held in a pool
  with a fixed number of slots and evicted in least-recently-used order.

For each request it computes the full-KV hit length, the usable prefix (the deepest
matching snapshot that lies within the KV hit), and from those the tokens that must
be recomputed. A policy decides where snapshots are taken during prefill and decode.

Resolution and conservative matching
------------------------------------
Traces identify prefixes by chained 64-token block hashes, so the simulator cannot see
equality inside a block. A snapshot at position ``p`` is keyed by the hash of the block
that contains token ``p - 1``; a later request can use it only if it has the same hash
at that block index. This never reports a false hit, but it can miss a real one when a
later request shares the first ``p`` tokens yet differs later in the same block. KV
coverage is recorded per block as the number of leading tokens cached, so a KV prefix
that ends inside a block is matched only by requests that share that whole block.

As in SGLang, a match covers at most ``prompt_tokens - 1`` tokens, so at least one token
is always computed.

Version 0 limitations
---------------------
* Requests run one at a time in arrival order; concurrency and in-flight slot
  reservations are not modeled.
* KV capacity and KV eviction are not modeled.
* Decode-time snapshots are matchable only through the request's first ``append``
  child, assuming the output is replayed token for token in the next request
  (``SimConfig.output_roundtrip``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Protocol

from ..traces.schema import BLOCK_TOKENS, Request, Trace


@dataclass(frozen=True)
class SimConfig:
    """Engine parameters that shape where snapshots can be taken."""

    cache_chunk_size: int = 64
    """Granularity at which a state can be taken: max(model chunk size, page size)."""
    state_chunk_size: int = 64
    """The model's own recurrent chunk size (``mamba_chunk_size``, else 64)."""
    page_size: int = 1
    chunked_prefill_size: int = 2048
    track_interval: int = 256
    """Decode snapshot interval (SGLang ``--mamba-track-interval``)."""
    state_slots: int | None = None
    """Snapshots the state pool can hold; ``None`` means unlimited."""
    output_roundtrip: bool = True
    """Whether a request's output reappears verbatim at the start of its append child."""

    @property
    def checkpoint_grid(self) -> int:
        return math.lcm(self.cache_chunk_size, self.page_size)

    @property
    def track_grid(self) -> int:
        return math.lcm(self.checkpoint_grid, self.track_interval)


class Policy(Protocol):
    """Decides where snapshots are inserted. Eviction is LRU in version 0."""

    name: str

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        """Positions inserted during prefill, which starts at ``start`` (the usable prefix)."""
        ...

    def plan_finish(self, req: Request, cfg: SimConfig) -> int | None:
        """Position inserted when the request completes, beyond those from prefill."""
        ...


@dataclass
class RequestRecord:
    req_id: str
    prompt_tokens: int
    kv_hit: int
    usable_prefix: int
    recompute_tokens: int
    retreat_tokens: int
    """``kv_hit - usable_prefix``: KV that was cached but could not be used."""
    snapshots_inserted: int


@dataclass
class SimResult:
    policy: str
    config: dict
    requests: int = 0
    prompt_tokens: int = 0
    recompute_tokens: int = 0
    retreat_tokens: int = 0
    snapshots_written: int = 0
    evictions: int = 0
    state_peak: int = 0
    records: list[RequestRecord] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("records")
        d["recompute_ratio"] = (
            round(self.recompute_tokens / self.prompt_tokens, 6) if self.prompt_tokens else 0.0
        )
        d["provenance"] = "simulated"
        return d


@dataclass
class _Snapshot:
    key: tuple[int, str]
    pos: int
    last_used: int


class _Store:
    """State snapshots indexed by (block count, block hash), with LRU eviction."""

    def __init__(self, slots: int | None) -> None:
        self.slots = slots
        self.by_key: dict[tuple[int, str], dict[int, _Snapshot]] = {}
        self.count = 0
        self.clock = 0
        self.evictions = 0
        self.peak = 0

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def insert(self, key: tuple[int, str], pos: int) -> bool:
        """Insert or refresh a snapshot. Returns True when a new snapshot was written."""
        bucket = self.by_key.setdefault(key, {})
        if pos in bucket:
            bucket[pos].last_used = self.tick()
            return False
        if self.slots is not None:
            if self.slots <= 0:
                return False
            while self.count >= self.slots:
                self._evict_lru()
        bucket[pos] = _Snapshot(key, pos, self.tick())
        self.count += 1
        self.peak = max(self.peak, self.count)
        return True

    def _evict_lru(self) -> None:
        victim = min(
            (s for b in self.by_key.values() for s in b.values()), key=lambda s: s.last_used
        )
        del self.by_key[victim.key][victim.pos]
        if not self.by_key[victim.key]:
            del self.by_key[victim.key]
        self.count -= 1
        self.evictions += 1

    def best_match(self, blocks: tuple[str, ...], limit: int) -> _Snapshot | None:
        """Deepest snapshot reachable through ``blocks`` at a position ``<= limit``."""
        best = None
        for k in range(1, len(blocks) + 1):
            if (k - 1) * BLOCK_TOKENS >= limit:
                break
            for s in self.by_key.get((k, blocks[k - 1]), {}).values():
                if s.pos <= limit and (best is None or s.pos > best.pos):
                    best = s
        return best


def _snapshot_key(blocks: tuple[str, ...], pos: int) -> tuple[int, str] | None:
    k = math.ceil(pos / BLOCK_TOKENS)
    if pos <= 0 or k > len(blocks):
        return None
    return (k, blocks[k - 1])


class _KV:
    """Cached KV prefixes: per (block index, block hash), the leading tokens present."""

    def __init__(self) -> None:
        self.covered: dict[tuple[int, str], int] = {}

    def insert(self, blocks: tuple[str, ...], upto: int) -> None:
        for k in range(1, min(len(blocks), math.ceil(upto / BLOCK_TOKENS)) + 1):
            n = min(BLOCK_TOKENS, upto - (k - 1) * BLOCK_TOKENS)
            key = (k, blocks[k - 1])
            if n > self.covered.get(key, 0):
                self.covered[key] = n

    def hit(self, req: Request) -> int:
        tokens = 0
        for k, h in enumerate(req.blocks, start=1):
            n = self.covered.get((k, h), 0)
            block_len = BLOCK_TOKENS if k < len(req.blocks) else req.last_block_tokens
            tokens += min(n, block_len)
            if n < block_len:
                break
        return tokens


def simulate(trace: Trace, policy: Policy, cfg: SimConfig) -> SimResult:
    """Replay ``trace`` under ``policy`` and return token-level outcomes."""
    store = _Store(cfg.state_slots)
    kv = _KV()
    first_append_child: dict[str, Request] = {}
    for r in trace.requests:
        if r.pattern == "append" and r.parent_req_id and r.parent_req_id not in first_append_child:
            first_append_child[r.parent_req_id] = r

    res = SimResult(policy=policy.name, config=asdict(cfg))

    def insert(blocks: tuple[str, ...], pos: int) -> int:
        key = _snapshot_key(blocks, pos)
        if key is None:
            return 0
        kv.insert(blocks, pos)
        return int(store.insert(key, pos))

    for _, r in trace.global_arrivals():
        kv_hit = min(kv.hit(r), max(r.prompt_tokens - 1, 0))

        snap = store.best_match(r.blocks, kv_hit)
        usable = 0
        if snap is not None:
            usable = snap.pos
            snap.last_used = store.tick()

        written = 0
        for pos in policy.plan_prefill(r, usable, kv_hit, cfg):
            written += insert(r.blocks, pos)

        end_pos = policy.plan_finish(r, cfg)
        if end_pos is not None:
            if end_pos <= r.prompt_tokens:
                written += insert(r.blocks, end_pos)
            elif cfg.output_roundtrip and r.req_id in first_append_child:
                child = first_append_child[r.req_id]
                if end_pos <= child.prompt_tokens:
                    written += insert(child.blocks, end_pos)

        res.records.append(
            RequestRecord(
                req_id=r.req_id,
                prompt_tokens=r.prompt_tokens,
                kv_hit=kv_hit,
                usable_prefix=usable,
                recompute_tokens=r.prompt_tokens - usable,
                retreat_tokens=kv_hit - usable,
                snapshots_inserted=written,
            )
        )
        res.requests += 1
        res.prompt_tokens += r.prompt_tokens
        res.recompute_tokens += r.prompt_tokens - usable
        res.retreat_tokens += kv_hit - usable
        res.snapshots_written += written

    res.evictions = store.evictions
    res.state_peak = store.peak
    return res


def sweep(
    trace: Trace, policies: Iterable[Policy], slots: Iterable[int | None], base: SimConfig
) -> list[dict]:
    """Run every policy at every state-pool size; one summary row per combination."""
    rows = []
    for p in policies:
        for n in slots:
            cfg = SimConfig(**{**asdict(base), "state_slots": n})
            rows.append(simulate(trace, p, cfg).summary())
    return rows

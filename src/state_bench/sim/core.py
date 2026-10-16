"""Trace-driven simulator for recurrent-state snapshot policies.

The simulator replays a trace request by request and tracks two caches:

* **KV coverage**: which prefixes have attention KV cached. Version 0 assumes KV
  capacity is unlimited, so coverage only grows.
* **State snapshots**: recurrent-state checkpoints at token positions, held in a pool
  with a fixed number of slots.

For each request it computes the full-KV hit length, the usable prefix (the deepest
matching snapshot that lies within the KV hit), and from those the tokens that must
be recomputed. A policy decides where snapshots are taken, which snapshot is evicted
when the pool is full, and whether a new snapshot is admitted at all.

Matching
--------
Traces identify prefixes by chained 64-token block hashes, so equality inside a block is
not visible from hashes alone. Two sources of prefix equality are used:

* **Full blocks.** Equal hashes at block index ``k`` mean the first ``64 * k`` tokens are
  equal. A snapshot at ``p`` whose block ``ceil(p / 64)`` is full in the request that
  stored it is keyed by that block's hash.
* **Lineage.** An ``append`` request starts with its parent's whole prompt, and a
  ``retry`` repeats it. A snapshot in the partial last block of a prompt is keyed by the
  request that stored it and can be used by requests that descend from it through
  ``append`` and ``retry`` links.

Prefix sharing that neither source shows, such as a fork that diverges inside a block,
is not credited. The simulator can therefore miss reuse that an engine would find; it
does not report reuse that is not there.

As in SGLang, a match covers at most ``prompt_tokens - 1`` tokens, so at least one token
is always computed.

Version 0 limitations
---------------------
* Requests run one at a time in arrival order; concurrency and in-flight slot
  reservations are not modeled.
* KV capacity and KV eviction are not modeled.
* A snapshot taken during decode is stored only if the request has an ``append`` child,
  assuming the reply reappears token for token at the start of that child
  (``SimConfig.output_roundtrip``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Iterable
from dataclasses import asdict, dataclass, field

from ..traces.schema import BLOCK_TOKENS, Request, Trace

LINK_PATTERNS = ("append", "retry")
"""Patterns whose request starts with the parent's whole prompt."""


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
    first_chunk_cut: str = "none"
    """``"none"``: chunks start at the matched prefix and take ``chunked_prefill_size``.
    ``"hashed"``: when a prompt needs more than one chunk, the first chunk is cut at a
    length in ``[1, chunked_prefill_size]`` derived from the request id, standing in for
    a shared prefill budget running out (an assumption, not a model of the scheduler)."""
    max_states_per_path: int | None = None
    """Per-path snapshot cap (SGLang ``--mamba-max-states-per-path``); ``None`` means
    unlimited. After each new snapshot, the shallowest snapshots on its path beyond the
    cap are evicted, skipping the new one, fork points, and the one the current request
    resumed from."""
    path_cap_min_usage: float = 0.0
    """Apply the per-path cap only while the state pool is at least this full (a
    fraction of ``state_slots``). ``0`` applies it always; with an unlimited pool, any
    positive value never applies it. Not an SGLang option: models a pressure-gated cap."""

    def __post_init__(self) -> None:
        if self.first_chunk_cut not in ("none", "hashed"):
            raise ValueError(
                f"first_chunk_cut must be 'none' or 'hashed', not {self.first_chunk_cut!r}"
            )
        if self.max_states_per_path is not None and self.max_states_per_path < 1:
            raise ValueError("max_states_per_path must be >= 1 or None")
        if not 0.0 <= self.path_cap_min_usage <= 1.0:
            raise ValueError("path_cap_min_usage must be in [0, 1]")

    @property
    def checkpoint_grid(self) -> int:
        return math.lcm(self.cache_chunk_size, self.page_size)

    @property
    def track_grid(self) -> int:
        return math.lcm(self.checkpoint_grid, self.track_interval)


def chunk_bounds(req: Request, start: int, cfg: SimConfig) -> list[tuple[int, int]]:
    """Prefill chunks ``[pos, end)`` for ``req`` resuming at ``start``."""
    out = []
    pos = start
    size = cfg.chunked_prefill_size
    if cfg.first_chunk_cut == "hashed" and req.prompt_tokens - start > size:
        digest = hashlib.blake2b(req.req_id.encode(), digest_size=8).digest()
        first = 1 + int.from_bytes(digest, "big") % size
        out.append((pos, pos + first))
        pos += first
    while pos < req.prompt_tokens:
        end = min(pos + size, req.prompt_tokens)
        out.append((pos, end))
        pos = end
    return out


@dataclass
class Snapshot:
    """A stored state. ``key`` identifies the prefix it belongs to (see module docs)."""

    key: Hashable
    pos: int
    owner: str
    """Request whose token sequence the snapshot was keyed against."""
    chain: str
    """Conversation chain: the root of the ``append``/``retry`` links leading to the owner."""
    created: int
    last_used: int
    """Logical clock values; they order events within a run."""
    last_used_s: float = 0.0
    """Arrival time of the request that created or last used the snapshot."""
    hits: int = 0
    """Requests that resumed from this snapshot."""
    branch_hits: int = 0
    """Of those, requests from a different conversation chain than the snapshot's."""


class Policy:
    """Base policy: no snapshots, least-recently-used eviction.

    Subclasses override placement (``plan_prefill``, ``plan_finish``), eviction order
    (``priority``), or both. ``bind`` is called once per run before any request.
    """

    name = "base"

    def bind(self, sim: Simulation) -> None:
        """Called before the first request; future-aware policies precompute here."""

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        """Positions stored during prefill, which starts at ``start`` (the usable prefix)."""
        return []

    def plan_finish(self, req: Request, cfg: SimConfig) -> list[int]:
        """Positions stored when the request completes. Positions past the prompt lie in
        the reply and are stored against the request's first ``append`` child."""
        return []

    def priority(self, snap: Snapshot, sim: Simulation) -> tuple:
        """Eviction order: the lowest value is evicted first. A new snapshot whose
        priority is the lowest in a full pool is not admitted."""
        return (snap.last_used,)


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
    snapshots_rejected: int = 0
    """Snapshots the policy declined to admit into a full pool."""
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


class Simulation:
    """State of one run. Policies read it through ``bind`` and ``priority``."""

    def __init__(self, trace: Trace, policy: Policy, cfg: SimConfig) -> None:
        self.trace = trace
        self.policy = policy
        self.cfg = cfg
        arrivals = trace.global_arrivals()
        self.order = [r for _, r in arrivals]
        self.arrival_s = [t for t, _ in arrivals]
        """Absolute arrival time of each request, by arrival index."""
        self.chain_last_arrival_s: dict[str, float] = {}
        """Arrival time of the latest request seen so far in each chain."""
        self.pending: Snapshot | None = None
        """A snapshot being considered for admission into a full pool."""
        self.by_id = {r.req_id: r for r in trace.requests}
        self.index = {r.req_id: i for i, r in enumerate(self.order)}
        self.now = -1
        """Arrival index of the request being processed."""

        self.first_append_child: dict[str, Request] = {}
        for r in self.order:
            if (
                r.pattern == "append"
                and r.parent_req_id
                and r.parent_req_id not in self.first_append_child
            ):
                self.first_append_child[r.parent_req_id] = r
        self.lineage: dict[str, tuple[str, ...]] = {}
        self.chain: dict[str, str] = {}
        for r in self.order:
            parent = self.by_id.get(r.parent_req_id) if r.parent_req_id else None
            if parent is not None and r.pattern in LINK_PATTERNS:
                self.lineage[r.req_id] = (r.req_id, *self.lineage[parent.req_id])
                self.chain[r.req_id] = self.chain[parent.req_id]
            else:
                self.lineage[r.req_id] = (r.req_id,)
                self.chain[r.req_id] = r.req_id

        self._kv_blocks: dict[tuple[int, str], int] = {}
        self._kv_children: dict[tuple[int, str], set[str]] = {}
        """Distinct next blocks seen after each cached full block (for fork points)."""
        self._kv_tail: dict[str, int] = {}
        self.locked: Snapshot | None = None
        """The snapshot the current request resumed from."""
        self._snaps: dict[Hashable, dict[int, Snapshot]] = {}
        self.count = 0
        self.clock = 0
        self.result = SimResult(policy=policy.name, config=asdict(cfg))

    @property
    def now_s(self) -> float:
        """Arrival time of the request being processed."""
        return self.arrival_s[self.now] if self.now >= 0 else 0.0

    # -- keys -----------------------------------------------------------------------

    @staticmethod
    def snapshot_key(owner: Request, pos: int) -> Hashable | None:
        """Key of a snapshot at ``pos`` in ``owner``'s token sequence."""
        k = math.ceil(pos / BLOCK_TOKENS)
        if pos <= 0 or pos > owner.prompt_tokens:
            return None
        if k <= len(owner.full_blocks):
            return (k, owner.blocks[k - 1])
        return ("tail", owner.req_id)

    # -- KV ---------------------------------------------------------------------------

    def _kv_insert(self, owner: Request, upto: int) -> None:
        for k in range(1, min(len(owner.blocks), math.ceil(upto / BLOCK_TOKENS))):
            self._kv_children.setdefault((k, owner.blocks[k - 1]), set()).add(owner.blocks[k])
        n_full = len(owner.full_blocks)
        for k in range(1, min(n_full, upto // BLOCK_TOKENS) + 1):
            self._kv_blocks[(k, owner.blocks[k - 1])] = BLOCK_TOKENS
        rem = upto - (upto // BLOCK_TOKENS) * BLOCK_TOKENS
        k = upto // BLOCK_TOKENS + 1
        if rem and k <= n_full:
            key = (k, owner.blocks[k - 1])
            self._kv_blocks[key] = max(self._kv_blocks.get(key, 0), rem)
        if upto > n_full * BLOCK_TOKENS:
            self._kv_tail[owner.req_id] = max(self._kv_tail.get(owner.req_id, 0), upto)

    def kv_hit(self, req: Request) -> int:
        tokens = 0
        for k, h in enumerate(req.full_blocks, start=1):
            n = self._kv_blocks.get((k, h), 0)
            tokens += n
            if n < BLOCK_TOKENS:
                break
        for rid in self.lineage[req.req_id]:
            tokens = max(tokens, self._kv_tail.get(rid, 0))
        return min(tokens, max(req.prompt_tokens - 1, 0))

    # -- snapshots ----------------------------------------------------------------------

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def snapshots(self) -> Iterable[Snapshot]:
        for bucket in self._snaps.values():
            yield from bucket.values()

    def best_match(self, req: Request, limit: int) -> Snapshot | None:
        """Deepest snapshot on ``req``'s prefix at a position ``<= limit``."""
        keys: list[Hashable] = []
        for k, h in enumerate(req.full_blocks, start=1):
            if (k - 1) * BLOCK_TOKENS >= limit:
                break
            keys.append((k, h))
        keys += [("tail", rid) for rid in self.lineage[req.req_id]]
        best = None
        for key in keys:
            for s in self._snaps.get(key, {}).values():
                if s.pos <= limit and (best is None or s.pos > best.pos):
                    best = s
        return best

    def store(self, owner: Request, pos: int, chain: str) -> int:
        """Store a snapshot at ``pos`` of ``owner``'s sequence. Returns 1 if newly written."""
        key = self.snapshot_key(owner, pos)
        if key is None:
            return 0
        bucket = self._snaps.get(key, {})
        if pos in bucket:
            bucket[pos].last_used = self.tick()
            bucket[pos].last_used_s = self.now_s
            self._kv_insert(owner, pos)
            return 0
        t = self.tick()
        snap = Snapshot(key, pos, owner.req_id, chain, t, t, self.now_s)
        slots = self.cfg.state_slots
        if slots is not None and self.count >= slots:
            if slots <= 0:
                self.result.snapshots_rejected += 1
                return 0
            self.pending = snap
            try:
                victim = min(self.snapshots(), key=lambda s: self.policy.priority(s, self))
                reject = self.policy.priority(snap, self) < self.policy.priority(victim, self)
            finally:
                self.pending = None
            if reject:
                self.result.snapshots_rejected += 1
                return 0
            self._evict(victim)
        self._snaps.setdefault(key, {})[pos] = snap
        self._kv_insert(owner, pos)
        self.count += 1
        self.result.state_peak = max(self.result.state_peak, self.count)
        if self.cfg.max_states_per_path is not None and self._cap_active():
            self._cap_path(owner, snap)
        return 1

    def _cap_active(self) -> bool:
        need = self.cfg.path_cap_min_usage
        if need <= 0:
            return True
        slots = self.cfg.state_slots
        return slots is not None and slots > 0 and self.count >= need * slots

    def _is_fork(self, snap: Snapshot) -> bool:
        """Whether more than one cached sequence continues past ``snap``.

        Known only at block boundaries; a snapshot inside a block or on a lineage tail
        is treated as having a single continuation.
        """
        key = snap.key
        if key[0] == "tail" or snap.pos != key[0] * BLOCK_TOKENS:
            return False
        return len(self._kv_children.get(key, ())) > 1

    def _cap_path(self, owner: Request, tail: Snapshot) -> None:
        """SGLang's per-path cap (R15): evict the shallowest eligible snapshots on the
        path to ``tail`` beyond ``max_states_per_path``."""
        keys: list[Hashable] = [
            (k, owner.blocks[k - 1])
            for k in range(1, min(len(owner.full_blocks), math.ceil(tail.pos / BLOCK_TOKENS)) + 1)
        ]
        keys += [("tail", rid) for rid in self.lineage[owner.req_id]]
        holders = sorted(
            (s for key in keys for s in self._snaps.get(key, {}).values() if s.pos <= tail.pos),
            key=lambda s: s.pos,
        )
        excess = len(holders) - self.cfg.max_states_per_path
        for s in holders:
            if excess <= 0 or s is tail:
                break
            if s is self.locked or self._is_fork(s):
                continue
            self._evict(s)
            excess -= 1

    def _evict(self, victim: Snapshot) -> None:
        bucket = self._snaps[victim.key]
        del bucket[victim.pos]
        if not bucket:
            del self._snaps[victim.key]
        self.count -= 1
        self.result.evictions += 1

    # -- run ----------------------------------------------------------------------------

    def run(self) -> SimResult:
        self.policy.bind(self)
        cfg, res = self.cfg, self.result
        for i, r in enumerate(self.order):
            self.now = i
            self.chain_last_arrival_s[self.chain[r.req_id]] = self.arrival_s[i]
            kv_hit = self.kv_hit(r)
            snap = self.best_match(r, kv_hit)
            self.locked = snap
            usable = 0
            if snap is not None:
                usable = snap.pos
                snap.last_used = self.tick()
                snap.last_used_s = self.now_s
                snap.hits += 1
                if self.chain[r.req_id] != snap.chain:
                    snap.branch_hits += 1

            chain = self.chain[r.req_id]
            written = 0
            for pos in self.policy.plan_prefill(r, usable, kv_hit, cfg):
                if pos > usable:
                    written += self.store(r, pos, chain)
            child = self.first_append_child.get(r.req_id)
            for pos in self.policy.plan_finish(r, cfg):
                if pos <= r.prompt_tokens:
                    written += self.store(r, pos, chain)
                elif cfg.output_roundtrip and child is not None:
                    written += self.store(child, pos, chain)

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
        return res


def simulate(trace: Trace, policy: Policy, cfg: SimConfig) -> SimResult:
    """Replay ``trace`` under ``policy`` and return token-level outcomes."""
    return Simulation(trace, policy, cfg).run()


def sweep(
    trace: Trace, policies: Iterable[Policy], slots: Iterable[int | None], base: SimConfig
) -> list[dict]:
    """Run every policy at every state-pool size; one summary row per combination."""
    rows = []
    slots = list(slots)
    for p in policies:
        for n in slots:
            cfg = SimConfig(**{**asdict(base), "state_slots": n})
            rows.append(simulate(trace, p, cfg).summary())
    return rows

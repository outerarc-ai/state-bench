"""Snapshots at message boundaries, admitted and evicted by estimated value.

Placement: during prefill, a snapshot at the end of each message in the prompt (system
prompt, user messages, assistant replies, tool results), rounded down to the checkpoint
grid; with ``reply_end=True``, also one at the end of the reply, taken during decode and
usable only when the reply round-trips into the next request. Several snapshots can come
from one prefill chunk, which assumes the engine can capture interior states of a forward
pass, as ATOM #2045 does; SGLang currently records at most one per chunk. With ``demand=True``, the last grid point of a KV hit that runs
past the usable state is also stored. Requests without message boundaries in the trace
get a snapshot at the prompt end only.

Value: each snapshot is scored

    value = p_reuse * pos * exp(-idle_s / horizon_s)

where ``pos`` is the prompt tokens a hit saves, ``idle_s`` the time since the snapshot
was created or last used, ``horizon_s`` is ``gap_factor`` times the mean gap between
requests seen so far in the snapshot's session (``tau_s`` until a gap has been seen), and
``p_reuse`` an estimate from the session's history so far:

* the latest turn's prompt-end and reply-end snapshots of a conversation chain are
  reused when the conversation continues, so ``p = (appends + retries + 1) / (requests + 2)``.
  Both stay live because the engine cannot tell in advance whether the reply will
  reappear token for token in the next request;
* any other snapshot, including end-of-turn snapshots of earlier turns of the chain, is
  reused only when a request branches there, so
  ``p = (forks + rewrites + 1) / (requests + 2)``;
* a snapshot that has served ``h`` requests from other chains (branches or other
  sessions) gets ``p = 1 - (1 - p) / (h + 1)``; use by the chain's own next request is
  expected and is not counted.

All snapshots of one model have the same size, so ranking by value equals ranking by
value per byte. The lowest value is evicted first, and a new snapshot worth less than
every resident one is not admitted. ``eviction="lru"`` keeps the placement and evicts
the least recently used snapshot instead, admitting every new one; it isolates what the
value ranking contributes. ``gap_factor``, ``tau_s``, and the estimators are
assumptions; the policy uses only requests that have already arrived.
"""

from __future__ import annotations

import math
from collections import Counter

from ...traces.schema import Request
from ..core import Policy, SimConfig, Simulation, Snapshot


class BoundaryValue(Policy):
    name = "boundary_value"

    def __init__(
        self,
        gap_factor: float = 3.0,
        tau_s: float = 120.0,
        demand: bool = False,
        reply_end: bool = True,
        eviction: str = "value",
    ) -> None:
        if tau_s <= 0 or gap_factor <= 0:
            raise ValueError("gap_factor and tau_s must be > 0")
        if eviction not in ("value", "lru"):
            raise ValueError("eviction must be 'value' or 'lru'")
        self.eviction = eviction
        self.gap_factor = gap_factor
        self.tau_s = tau_s
        self.demand = demand
        self.reply_end = reply_end
        self._sim: Simulation | None = None
        self._session_requests: Counter[str] = Counter()
        self._session_patterns: dict[str, Counter[str]] = {}
        self._live_ends: dict[str, set[tuple[str, int]]] = {}
        self._last_arrival: dict[str, float] = {}
        self._gaps: dict[str, tuple[float, int]] = {}

    def bind(self, sim: Simulation) -> None:
        self._sim = sim
        self._session_requests = Counter()
        self._session_patterns = {}
        self._live_ends = {}
        self._last_arrival = {}
        self._gaps = {}

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        now = self._sim.now_s
        last = self._last_arrival.get(req.session_id)
        if last is not None:
            total, n = self._gaps.get(req.session_id, (0.0, 0))
            self._gaps[req.session_id] = (total + now - last, n + 1)
        self._last_arrival[req.session_id] = now
        self._session_requests[req.session_id] += 1
        if req.pattern:
            self._session_patterns.setdefault(req.session_id, Counter())[req.pattern] += 1
        grid = cfg.checkpoint_grid
        end = (req.prompt_tokens // grid) * grid
        self._live_ends[self._sim.chain[req.req_id]] = {(req.req_id, end)}
        positions = {end}
        if req.segments:
            positions.update((seg.end_tok // grid) * grid for seg in req.segments)
        if self.demand:
            positions.add((kv_hit // grid) * grid)
        return sorted(p for p in positions if start < p <= req.prompt_tokens)

    def plan_finish(self, req: Request, cfg: SimConfig) -> list[int]:
        if not self.reply_end:
            return []
        grid = cfg.checkpoint_grid
        pos = ((req.prompt_tokens + req.output_tokens - 1) // grid) * grid
        if pos <= req.prompt_tokens:
            return []
        # The simulator keys a reply snapshot against the next request, which it stores
        # only if that request exists; the policy itself does not look ahead.
        child = self._sim.first_append_child.get(req.req_id)
        owner = child.req_id if child is not None else f"{req.req_id}:reply"
        self._live_ends.setdefault(self._sim.chain[req.req_id], set()).add((owner, pos))
        return [pos]

    def reuse_probability(self, snap: Snapshot, sim: Simulation) -> float:
        session = sim.by_id[snap.owner].session_id
        n = self._session_requests[session]
        pat = self._session_patterns.get(session, Counter())
        if (snap.owner, snap.pos) in self._live_ends.get(snap.chain, ()):
            hits = pat["append"] + pat["retry"]
        else:
            hits = pat["fork"] + pat["rewrite"]
        p = (hits + 1) / (n + 2)
        return 1 - (1 - p) / (snap.branch_hits + 1)

    def value(self, snap: Snapshot, sim: Simulation) -> float:
        idle = max(sim.now_s - snap.last_used_s, 0.0)
        total, n = self._gaps.get(sim.by_id[snap.owner].session_id, (0.0, 0))
        horizon = self.gap_factor * total / n if n and total > 0 else self.tau_s
        return self.reuse_probability(snap, sim) * snap.pos * math.exp(-idle / horizon)

    def priority(self, snap: Snapshot, sim: Simulation) -> tuple:
        if self.eviction == "lru":
            return (snap.last_used,)
        return (self.value(snap, sim), snap.last_used)

"""Offline reference policy that knows every future request.

For each request, the oracle finds the deepest position at which any earlier request's
state could serve it, given what the simulator can verify (shared full blocks and
``append``/``retry`` lineage, see ``sim/core.py``). It stores a snapshot at exactly those
positions, during prefill or decode, whenever the request that computes the position
runs. When the pool is full it evicts the snapshot whose next use is furthest in the
future (Belady's rule) and does not admit a snapshot that would be that victim.

With an unlimited pool the result is the least recomputation any policy can reach in
this simulator. With a limited pool the result is a strong offline reference rather than
a proven optimum, because snapshots differ in the tokens they save.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Hashable

from ...traces.schema import BLOCK_TOKENS, Request
from ..core import Policy, SimConfig, Simulation, Snapshot


class OfflineOracle(Policy):
    name = "offline_oracle"

    def __init__(self) -> None:
        self._sim: Simulation | None = None
        self._uses: dict[tuple[Hashable, int], list[int]] = {}
        self._positions: dict[Hashable, set[int]] = {}
        self._finish: dict[str, set[int]] = {}

    def bind(self, sim: Simulation) -> None:
        self._sim = sim
        self._uses = {}
        self._positions = {}
        self._finish = {}
        roundtrip = sim.cfg.output_roundtrip
        seen: set[tuple[int, str]] = set()
        for i, c in enumerate(sim.order):
            # Deepest full block shared with any earlier request.
            k_star = 0
            for k, h in enumerate(c.full_blocks, start=1):
                if (k, h) not in seen:
                    break
                k_star = k
            best: tuple[int, Request, Request | None] | None = None
            if k_star:
                best = (k_star * BLOCK_TOKENS, sim.by_id[c.req_id], None)
            # Ancestors whose whole prompt (and reply, if it round-trips) c starts with.
            lin = sim.lineage[c.req_id]
            for succ_id, anc_id in itertools.pairwise(lin):
                anc = sim.by_id[anc_id]
                succ = sim.by_id[succ_id]
                if best is None or anc.prompt_tokens > best[0]:
                    best = (anc.prompt_tokens, anc, None)
                child = sim.first_append_child.get(anc_id)
                reply_end = anc.prompt_tokens + anc.output_tokens - 1
                if (
                    roundtrip
                    and child is succ
                    and reply_end > anc.prompt_tokens
                    and reply_end > best[0]
                ):
                    best = (reply_end, child, anc)
            seen.update((k, h) for k, h in enumerate(c.full_blocks, start=1))
            if best is None:
                continue
            pos, owner, emitter = best
            pos = min(pos, c.prompt_tokens - 1)
            if pos <= 0:
                continue
            key = Simulation.snapshot_key(owner, pos)
            if key is None:  # pragma: no cover - positions above are inside owner's prompt
                continue
            self._uses.setdefault((key, pos), []).append(i)
            if emitter is None:
                self._positions.setdefault(key, set()).add(pos)
            else:
                self._finish.setdefault(emitter.req_id, set()).add(pos)

    def _needed(self, key: Hashable, pos: int) -> bool:
        uses = self._uses.get((key, pos), [])
        return bisect.bisect_right(uses, self._sim.now) < len(uses)

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        keys: list[Hashable] = [(k, h) for k, h in enumerate(req.full_blocks, start=1)]
        keys.append(("tail", req.req_id))
        out = []
        for key in keys:
            for pos in self._positions.get(key, ()):
                if (
                    start < pos <= req.prompt_tokens
                    and Simulation.snapshot_key(req, pos) == key
                    and self._needed(key, pos)
                ):
                    out.append(pos)
        return sorted(out)

    def plan_finish(self, req: Request, cfg: SimConfig) -> list[int]:
        child = self._sim.first_append_child.get(req.req_id)
        if child is None:
            return []
        return sorted(
            p
            for p in self._finish.get(req.req_id, ())
            if self._needed(Simulation.snapshot_key(child, p), p)
        )

    def priority(self, snap: Snapshot, sim: Simulation) -> tuple:
        uses = self._uses.get((snap.key, snap.pos), [])
        j = bisect.bisect_right(uses, sim.now)
        if j == len(uses):
            return (float("-inf"), snap.last_used)
        return (-uses[j], snap.last_used)

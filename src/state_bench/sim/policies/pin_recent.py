"""Retain the most recent snapshots of active conversations.

Placement follows SGLang's current rules. Eviction protects the ``k`` most recently
created snapshots of each active chain, where a chain is a conversation linked by
``append`` and ``retry`` requests and is active while its latest request arrived within
``active_s`` seconds. Protected snapshots are evicted, least recently used first, only
when every resident snapshot is protected.

This is the retention policy proposed in SGLang RFC #40865 (closed by its author), which
suggests pinning the most recent K checkpoints per active session. How an engine would
detect an active session is not specified there; ``active_s`` is an assumption.
"""

from __future__ import annotations

from ..core import Simulation, Snapshot
from .sglang_current import SGLangCurrent


class PinRecent(SGLangCurrent):
    name = "pin_recent"

    def __init__(self, k: int = 4, active_s: float | None = 120.0) -> None:
        super().__init__()
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self.active_s = active_s
        self._stamp: tuple[int, int, int] | None = None
        self._pinned: set[int] = set()

    def bind(self, sim: Simulation) -> None:
        self._stamp = None
        self._pinned = set()

    def _active(self, chain: str, sim: Simulation) -> bool:
        if self.active_s is None:
            return True
        return sim.arrival_s[sim.now] - sim.chain_last_arrival_s[chain] <= self.active_s

    def _refresh(self, sim: Simulation) -> None:
        stamp = (sim.clock, sim.count, id(sim.pending))
        if stamp == self._stamp:
            return
        by_chain: dict[str, list[Snapshot]] = {}
        candidates = list(sim.snapshots())
        if sim.pending is not None:
            candidates.append(sim.pending)
        for s in candidates:
            by_chain.setdefault(s.chain, []).append(s)
        pinned: set[int] = set()
        for chain, snaps in by_chain.items():
            if self.k and self._active(chain, sim):
                snaps.sort(key=lambda s: s.created, reverse=True)
                pinned.update(id(s) for s in snaps[: self.k])
        self._pinned = pinned
        self._stamp = stamp

    def priority(self, snap: Snapshot, sim: Simulation) -> tuple:
        self._refresh(sim)
        return (int(id(snap) in self._pinned), snap.last_used)

"""Hand-checked cases for the ``boundary_value`` policy (message boundaries, value-based
admission and eviction). Defaults: checkpoint_grid 64."""

import math

import pytest
from sim_helpers import make_trace, run, toks

from state_bench.sim import SimConfig, Simulation, Snapshot
from state_bench.sim.policies import BoundaryValue
from state_bench.traces.schema import Segment


def bound(trace, policy, **cfg) -> Simulation:
    sim = Simulation(trace, policy, SimConfig(**cfg))
    policy.bind(sim)
    sim.now = 0
    return sim


def test_positions_are_message_boundaries():
    segs = (Segment("system", 0, 100), Segment("user", 100, 150))
    trace = make_trace(("A", toks(1, 150), 1, None, None), segments=[segs])
    a = trace.requests[0]
    p = BoundaryValue()
    sim = bound(trace, p)
    # Ends at 100 and 150, rounded down to the grid: 64 and 128 (the prompt end).
    assert p.plan_prefill(a, 0, 0, sim.cfg) == [64, 128]
    assert p.plan_prefill(a, 64, 0, sim.cfg) == [128]
    # Without boundaries in the trace: the prompt end only.
    plain = make_trace(("A", toks(1, 150), 1, None, None)).requests[0]
    assert p.plan_prefill(plain, 0, 0, sim.cfg) == [128]


def test_demand_adds_kv_hit_point():
    trace = make_trace(("A", toks(1, 300), 1, None, None))
    p = BoundaryValue(demand=True)
    sim = bound(trace, p)
    assert p.plan_prefill(trace.requests[0], 0, 200, sim.cfg) == [192, 256]


def test_reply_end():
    trace = make_trace(("A", toks(1, 200), 100, None, None), ("B", toks(1, 200), 10, None, None))
    p = BoundaryValue()
    sim = bound(trace, p)
    a, b = trace.requests
    # A computes 299 tokens: reply end 256. B computes 209: 192 is inside its prompt.
    assert p.plan_finish(a, sim.cfg) == [256]
    assert p.plan_finish(b, sim.cfg) == []
    assert BoundaryValue(reply_end=False).plan_finish(a, sim.cfg) == []


def test_superseded_end_is_valued_as_a_branch_point():
    a = toks(1, 200)
    trace = make_trace(
        ("A1", a, 1, None, None),
        ("A2", a + toks(2, 60), 1, "A1", "append"),
        ("Z", toks(9, 100), 1, None, None),
        times=[0, 1, 20],
        sessions=["s", "s", "z"],
    )
    p = BoundaryValue()
    sim = Simulation(trace, p, SimConfig())
    sim.run()
    snaps = {(s.owner, s.pos): s for s in sim.snapshots()}
    old, new = snaps[("A1", 192)], snaps[("A2", 256)]
    # Session s: 2 requests, 1 append, no branches. A2's end is the chain's newest:
    # p = (1 + 1) / (2 + 2). A1's end is superseded: p = (0 + 1) / (2 + 2). A2's use of
    # A1's end is the chain's own continuation, so it does not raise p.
    assert (old.hits, old.branch_hits) == (1, 0)
    assert p.reuse_probability(new, sim) == pytest.approx(0.5)
    assert p.reuse_probability(old, sim) == pytest.approx(0.25)
    # Both were last touched at t=1; now is t=20. Session s has one 1 s gap, so the
    # horizon is 3 * 1 s.
    decay = math.exp(-19 / 3)
    assert p.value(new, sim) == pytest.approx(0.5 * 256 * decay)
    assert p.value(old, sim) == pytest.approx(0.25 * 192 * decay)
    # Session z has no gap yet: horizon tau_s, and its snapshot is fresh.
    z = snaps[("Z", 64)]
    assert p.value(z, sim) == pytest.approx(1 / 3 * 64)


def test_branch_hits_raise_reuse_probability():
    shared = toks(1, 128)
    trace = make_trace(
        ("A", shared, 1, None, None),
        ("B", shared + toks(2, 64), 1, None, None),
        sessions=["a", "b"],
    )
    p = BoundaryValue()
    sim = Simulation(trace, p, SimConfig())
    sim.run()
    snaps = {(s.owner, s.pos): s for s in sim.snapshots()}
    a = snaps[("A", 128)]
    # B (another chain) shares A's whole prompt and resumes from A's prompt end at 128.
    assert (a.hits, a.branch_hits) == (1, 1)
    # A's end is still the newest of A's chain; session a: 1 request, no appends:
    # p = 1/3, then one branch hit: 1 - (2/3) / 2.
    assert p.reuse_probability(a, sim) == pytest.approx(2 / 3)


def test_dead_long_session_is_evicted_before_live_short_one():
    long_ = toks(1, 1000)
    short = toks(2, 200)
    trace = make_trace(
        ("L1", long_, 1, None, None),
        ("L2", long_ + toks(3, 30), 1, "L1", "append"),
        ("S1", short, 1, None, None),
        ("S2", short + toks(4, 30), 1, "S1", "append"),
        ("S3", short + toks(4, 30) + toks(5, 30), 1, "S2", "append"),
        times=[0, 1, 30, 31, 32],
        sessions=["l", "l", "s", "s", "s"],
    )
    res, rec = run(trace, BoundaryValue(), state_slots=1)
    # At t=30, L2's end (1024 tokens, idle 29 s against a 3 s horizon) is worth less
    # than S1's fresh end, so S1's end is admitted and S2 resumes from it.
    assert rec["S2"].usable_prefix == 192
    assert rec["S3"].usable_prefix == 192
    assert res.evictions >= 1


def test_invalid_parameters():
    with pytest.raises(ValueError):
        BoundaryValue(tau_s=0)
    with pytest.raises(ValueError):
        BoundaryValue(gap_factor=-1)


def test_prompt_end_and_reply_end_both_stay_live_until_next_turn():
    a = toks(1, 200)
    trace = make_trace(
        ("A1", a, 100, None, None),
        ("A2", a + toks(2, 100) + toks(3, 30), 1, "A1", "append"),
    )
    a1, a2 = trace.requests
    p = BoundaryValue()
    sim = bound(trace, p)
    chain = sim.chain["A1"]
    assert p.plan_prefill(a1, 0, 0, sim.cfg) == [192]
    assert p.plan_finish(a1, sim.cfg) == [256]
    prompt_end = Snapshot(sim.snapshot_key(a1, 192), 192, "A1", chain, 1, 1)
    reply_end = Snapshot(sim.snapshot_key(a2, 256), 256, "A2", chain, 2, 2)
    # One request in the session, no appends yet: p = 1/3 for both live ends.
    assert p.reuse_probability(prompt_end, sim) == pytest.approx(1 / 3)
    assert p.reuse_probability(reply_end, sim) == pytest.approx(1 / 3)
    # A2 arrives: its own prompt end (320) is now the only live end.
    sim.now = 1
    p.plan_prefill(a2, 256, 256, sim.cfg)
    assert p.reuse_probability(reply_end, sim) == pytest.approx(1 / 4)


def test_lru_eviction_option_keeps_placement_and_ranks_by_recency():
    segs = (Segment("system", 0, 100), Segment("user", 100, 150))
    trace = make_trace(("A", toks(1, 150), 1, None, None), segments=[segs])
    a = trace.requests[0]
    p = BoundaryValue(eviction="lru")
    sim = bound(trace, p)
    assert p.plan_prefill(a, 0, 0, sim.cfg) == [64, 128]
    old = Snapshot(("k", "x"), 64, a.req_id, a.req_id, created=1, last_used=1)
    new = Snapshot(("k", "y"), 128, a.req_id, a.req_id, created=2, last_used=5)
    assert p.priority(old, sim) == (1,)
    assert p.priority(old, sim) < p.priority(new, sim)


def test_rejects_unknown_eviction():
    with pytest.raises(ValueError):
        BoundaryValue(eviction="random")

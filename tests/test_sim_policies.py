"""Hand-checked cases for the comparison policies and the offline oracle.

Defaults: cache_chunk_size 64, checkpoint_grid 64, track_interval 256, chunked_prefill_size
2048. Request ``i`` arrives at ``i`` seconds.
"""

import json

import pytest
from sim_helpers import make_trace, req, run, toks

from state_bench import cli
from state_bench.sim import POLICIES, SimConfig, chunk_bounds, get_policy, simulate
from state_bench.sim.policies import (
    MarconiAdmission,
    OfflineOracle,
    PinRecent,
    PromptEndAnchor,
    SGLangCurrent,
)
from state_bench.traces import schema, synth


def forks_of_shared_192():
    shared = toks(1, 192)
    return make_trace(
        ("A", shared + toks(2, 108), 1, None, None),
        ("B", shared + toks(3, 108), 1, "A", "fork"),
        ("C", shared + toks(4, 108), 1, "A", "fork"),
    )


# --- chunking ------------------------------------------------------------------


def test_hashed_first_chunk_cut():
    r = req(toks(1, 5000))
    cfg = SimConfig(chunked_prefill_size=2048, first_chunk_cut="hashed")
    bounds = chunk_bounds(r, 0, cfg)
    first = bounds[0][1]
    assert 1 <= first <= 2048
    assert all(e - p == 2048 for p, e in bounds[1:-1])
    assert bounds[-1][1] == 5000
    assert chunk_bounds(r, 0, cfg) == bounds  # deterministic
    # A prompt that fits in one chunk is not cut.
    assert chunk_bounds(req(toks(1, 2000)), 0, cfg) == [(0, 2000)]


def test_invalid_first_chunk_cut():
    with pytest.raises(ValueError, match="first_chunk_cut"):
        SimConfig(first_chunk_cut="random")


# --- absolute donation mask --------------------------------------------------------


def test_absolute_mask_stops_donating_after_off_grid_chunk():
    cfg = SimConfig(chunked_prefill_size=100)
    a = req(toks(1, 320))
    # [0,100) stores 64; later chunks start off the grid: (192-100), (256-200), and
    # (320-300) are not multiples of 64.
    assert SGLangCurrent(absolute_mask=True).plan_prefill(a, 0, 0, cfg) == [64]
    assert SGLangCurrent().plan_prefill(a, 0, 0, cfg) == [64, 164, 264]
    # On-grid chunks donate under both masks.
    cfg = SimConfig(chunked_prefill_size=128)
    assert SGLangCurrent(absolute_mask=True).plan_prefill(a, 0, 0, cfg) == [128, 256, 320]


def test_absolute_mask_next_turn():
    a = toks(1, 320)
    trace = make_trace(("A", a, 1, None, None), ("B", a + toks(2, 50), 1, "A", "append"))
    _, rel = run(trace, SGLangCurrent(), chunked_prefill_size=100)
    _, ab = run(trace, get_policy("sglang_absolute_mask"), chunked_prefill_size=100)
    assert (rel["B"].usable_prefix, ab["B"].usable_prefix) == (264, 64)
    # R10: KV stops at the last snapshot, so the absolute mask also loses the KV.
    assert ab["B"].kv_hit == 64


# --- prompt-end anchor ------------------------------------------------------------


def test_prompt_end_anchor_one_snapshot_per_prompt():
    a = toks(1, 320)
    trace = make_trace(("A", a, 1, None, None), ("B", a + toks(2, 50), 1, "A", "append"))
    res, rec = run(trace, PromptEndAnchor(), chunked_prefill_size=100)
    # A stores only 320; B resumes there and stores 320 + 50 rounded down = 320 again
    # (not past its start), so nothing new.
    assert rec["A"].snapshots_inserted == 1
    assert (rec["B"].usable_prefix, rec["B"].snapshots_inserted) == (320, 0)
    assert res.snapshots_written == 1


def test_prompt_end_anchor_misses_forks():
    _, rec = run(forks_of_shared_192(), PromptEndAnchor())
    # Without a demand snapshot, no state ever lands at the shared 192.
    assert [rec[x].usable_prefix for x in "BC"] == [0, 0]
    assert [rec[x].retreat_tokens for x in "BC"] == [192, 192]


# --- Marconi-style admission ------------------------------------------------------


def test_marconi_last_state_in_reply():
    a = toks(1, 200)
    trace = make_trace(
        ("A", a, 100, None, None),
        ("B", a + toks(2, 100) + toks(3, 30), 1, "A", "append"),
    )
    _, rec = run(trace, MarconiAdmission())
    # A computes 299 tokens; its last state is at 256, inside the reply.
    assert rec["A"].snapshots_inserted == 1
    assert rec["B"].usable_prefix == 256
    # B's last state: 330 tokens computed, rounded down to 320, inside its prompt.
    assert MarconiAdmission().plan_finish(req(toks(1, 330)), SimConfig()) == [320]


def test_marconi_shared_prefix_on_second_hit():
    _, rec = run(forks_of_shared_192(), MarconiAdmission())
    # A stores its last state at 256. B's KV hit (192) runs past its usable state (0),
    # so B stores 192, which C then uses.
    assert (rec["B"].usable_prefix, rec["B"].snapshots_inserted) == (0, 2)
    assert rec["C"].usable_prefix == 192


# --- pin recent -------------------------------------------------------------------


def pin_trace():
    a = toks(1, 320)
    return make_trace(
        ("A", a, 1, None, None),  # chunks of 100: 64, 164, 264
        ("X", toks(9, 400), 1, None, None),  # 64, 164, 264, 364
        ("A2", a + toks(2, 50), 1, "A", "append"),
    )


def test_pin_recent_keeps_newest_snapshot_of_each_chain():
    _, lru = run(pin_trace(), SGLangCurrent(), chunked_prefill_size=100, state_slots=3)
    _, pin = run(
        pin_trace(),
        PinRecent(k=1, active_s=None),
        chunked_prefill_size=100,
        state_slots=3,
    )
    # LRU: X's four snapshots push out all of A's.
    assert lru["A2"].usable_prefix == 0
    # Pinned: A's 264 survives; X evicts A64, A164, then its own older snapshots.
    assert pin["A2"].usable_prefix == 264


def test_pin_recent_releases_inactive_chains():
    _, rec = run(
        pin_trace(),
        PinRecent(k=1, active_s=0.5),
        chunked_prefill_size=100,
        state_slots=3,
    )
    # A's chain last arrived 1 s before X, beyond active_s, so nothing is pinned.
    assert rec["A2"].usable_prefix == 0


def test_pin_recent_k0_is_lru():
    a = run(pin_trace(), PinRecent(k=0), chunked_prefill_size=100, state_slots=3)[0]
    b = run(pin_trace(), SGLangCurrent(), chunked_prefill_size=100, state_slots=3)[0]
    assert (a.recompute_tokens, a.evictions) == (b.recompute_tokens, b.evictions)
    with pytest.raises(ValueError):
        PinRecent(k=-1)


# --- offline oracle -----------------------------------------------------------------


def test_oracle_places_branch_points_in_advance():
    res, rec = run(forks_of_shared_192(), OfflineOracle())
    # A stores 192 because B and C will need it.
    assert [rec[x].usable_prefix for x in "ABC"] == [0, 192, 192]
    assert res.snapshots_written == 1


def test_oracle_uses_reply_end():
    a = toks(1, 200)
    trace = make_trace(
        ("A", a, 100, None, None),
        ("B", a + toks(2, 100) + toks(3, 30), 1, "A", "append"),
    )
    _, rec = run(trace, OfflineOracle())
    # A computes up to 299 tokens and stores that state for B.
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].recompute_tokens) == (299, 299, 31)
    _, rec = run(trace, OfflineOracle(), output_roundtrip=False)
    assert rec["B"].usable_prefix == 200


def test_oracle_exact_retry_stops_one_short():
    a = toks(1, 256)
    _, rec = run(make_trace(("A", a, 1, None, None), ("R", a, 1, "A", "retry")), OfflineOracle())
    assert (rec["R"].usable_prefix, rec["R"].recompute_tokens) == (255, 1)


def test_oracle_belady_bypass():
    a, x = toks(1, 200), toks(2, 200)
    trace = make_trace(
        ("A", a, 1, None, None),
        ("X", x, 1, None, None),
        ("A2", a + toks(3, 30), 1, "A", "append"),
        ("X2", x + toks(4, 30), 1, "X", "append"),
    )
    res, rec = run(trace, OfflineOracle(), state_slots=1)
    # X's snapshot is next used later than A's, so it is not admitted.
    assert res.snapshots_rejected == 1
    assert (rec["A2"].usable_prefix, rec["X2"].usable_prefix) == (200, 0)


def test_oracle_bounds_every_policy_with_unlimited_slots():
    tr = synth.generate_trace(synth.preset_params("mixed", seed=3, sessions=6))
    for cut in ("none", "hashed"):
        cfg = SimConfig(chunked_prefill_size=1000, first_chunk_cut=cut)
        best = simulate(tr, OfflineOracle(), cfg).recompute_tokens
        for name in POLICIES:
            assert best <= simulate(tr, get_policy(name), cfg).recompute_tokens, name


# --- registry and CLI ---------------------------------------------------------------


def test_policy_specs():
    p = get_policy("pin_recent:k=2,active_s=none")
    assert (p.k, p.active_s, p.name) == (2, None, "pin_recent:k=2,active_s=none")
    assert get_policy("pin_recent:active_s=30.5").active_s == 30.5
    assert get_policy("sglang_current").name == "sglang_current"
    with pytest.raises(ValueError, match="key=value"):
        get_policy("pin_recent:k")
    with pytest.raises(ValueError, match="bad parameters"):
        get_policy("pin_recent:z=true")
    with pytest.raises(ValueError, match="bad parameters"):
        get_policy("offline_oracle:mode=fast")


def test_cli_policies(tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    schema.dump_trace(synth.generate_trace(synth.preset_params("mixed", seed=2, sessions=3)), path)
    argv = ["sim", str(path), "--first-chunk-cut", "hashed", "--state-slots", "8"]
    for name in POLICIES:
        argv += ["--policy", name]
    assert cli.main(argv) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["policy"] for r in rows] == list(POLICIES)
    assert all(r["config"]["first_chunk_cut"] == "hashed" for r in rows)
    assert cli.main(["sim", str(path), "--policy", "nope"]) == 2

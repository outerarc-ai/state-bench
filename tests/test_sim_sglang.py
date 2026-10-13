"""Hand-checked cases for the simulator and the ``sglang_current`` policy.

Each case builds requests from explicit token lists, so every expected position can be
derived from the rules in docs/SGLANG_MAMBA_CHECKPOINTS.md. Defaults: cache_chunk_size
64, checkpoint_grid 64, track_interval 256, chunked_prefill_size 2048.
"""

import json

import pytest
from sim_helpers import make_trace, req, toks
from sim_helpers import run as sim_run

from state_bench import cli
from state_bench.sim import SimConfig, get_policy, simulate, sweep
from state_bench.sim.core import Simulation
from state_bench.sim.policies import SGLangCurrent
from state_bench.traces import schema, synth

POLICY = SGLangCurrent()


def run(trace, **cfg):
    return sim_run(trace, POLICY, **cfg)


def res_req(tokens, out=1):
    return req(tokens, out)


# --- prefill placement (R6, R8) ------------------------------------------------


def test_single_chunk_snapshot_and_append_reuse():
    a = toks(1, 200)
    b = a + toks(2, 20) + toks(3, 30)  # reply replayed, then a tool result
    res, rec = run(make_trace(("A", a, 20, None, None), ("B", b, 20, "A", "append")))
    assert POLICY.plan_prefill(res_req(a), 0, 0, SimConfig()) == [192]
    # B: KV and the state both stop at 192 (R10); 58 tokens remain, too few for a snapshot.
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].recompute_tokens) == (192, 192, 58)
    assert rec["B"].snapshots_inserted == 0
    assert res.snapshots_written == 1


def test_chunk_relative_snapshots_off_grid():
    a = toks(1, 320)
    b = a + toks(2, 50)
    # Chunks of 100 from 0: [0,100) -> 64, [100,200) -> 164, [200,300) -> 264, [300,320) none.
    assert POLICY.plan_prefill(res_req(a), 0, 0, SimConfig(chunked_prefill_size=100)) == [
        64,
        164,
        264,
    ]
    _, rec = run(
        make_trace(("A", a, 1, None, None), ("B", b, 1, "A", "append")),
        chunked_prefill_size=100,
    )
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].retreat_tokens) == (264, 264, 0)


def test_chunk_size_on_grid():
    # [0,128) -> 128, [128,256) -> 256, [256,320) -> 320.
    a = toks(1, 320)
    cfg = SimConfig(chunked_prefill_size=128)
    assert POLICY.plan_prefill(res_req(a), 0, 0, cfg) == [128, 256, 320]


def test_partial_block_snapshot_is_matched_through_lineage():
    # A's snapshot at 264 sits in A's partial last block (256..299). B appends to A, so it
    # starts with all of A's prompt and can use it, as SGLang would.
    a = toks(1, 300)
    b = a + toks(2, 50)
    _, rec = run(
        make_trace(("A", a, 1, None, None), ("B", b, 1, "A", "append")),
        chunked_prefill_size=100,
    )
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].retreat_tokens) == (264, 264, 0)


def test_fork_inside_a_block_is_matched_conservatively():
    # B shares A's first 280 tokens. Hashes show only the first 256, so the simulator
    # credits A's snapshot at 164 where SGLang would use 264.
    a = toks(1, 300)
    b = a[:280] + toks(3, 70)
    _, rec = run(
        make_trace(("A", a, 1, None, None), ("B", b, 1, "A", "fork")),
        chunked_prefill_size=100,
    )
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].retreat_tokens) == (256, 164, 92)


# --- branching point (R2, R7) --------------------------------------------------


def test_branching_point_replaces_default_snapshot():
    shared = toks(1, 192)
    a = shared + toks(2, 108)
    b = shared + toks(3, 108)
    c = shared + toks(4, 108)
    _, rec = run(
        make_trace(
            ("A", a, 1, None, None),
            ("B", b, 1, "A", "fork"),
            ("C", c, 1, "A", "fork"),
        )
    )
    # A inserts 256 only. B's KV hit is 192 but no state lies there.
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].retreat_tokens) == (192, 0, 192)
    # B's one chunk records the branching point 192 instead of 256.
    assert POLICY.plan_prefill(res_req(b), 0, 192, SimConfig()) == [192]
    assert (rec["C"].usable_prefix, rec["C"].recompute_tokens) == (192, 108)


def test_branching_point_dropped_when_misaligned_to_chunk_start():
    shared = toks(1, 192)
    a = shared + toks(2, 108)
    b = shared + toks(3, 108)
    _, rec = run(
        make_trace(("A", a, 1, None, None), ("B", b, 1, "A", "fork")),
        chunked_prefill_size=100,
    )
    # A: 64, 164, 264. B resumes at 164; its chunk [164,264) would need (192-164) % 64 == 0.
    assert (rec["B"].kv_hit, rec["B"].usable_prefix, rec["B"].retreat_tokens) == (192, 164, 28)
    assert POLICY.plan_prefill(res_req(b), 164, 192, SimConfig(chunked_prefill_size=100)) == [228]


def test_branching_limit_when_grid_is_not_a_model_chunk_boundary():
    # page_size 128 with a 64-token model chunk: cache_chunk_size 128, grid 128. Extend of
    # 300 gives grid track 256 but model-aligned 256 too, so the limit is the chunk end.
    cfg = SimConfig(cache_chunk_size=128, page_size=128, state_chunk_size=64)
    assert POLICY.plan_prefill(res_req(toks(1, 300)), 0, 299, cfg) == [256]
    # Extend of 200: grid track 128, model-aligned 192, so the limit is 129 and a
    # branching point at 128 still applies; it equals the default.
    assert POLICY.plan_prefill(res_req(toks(1, 200)), 0, 199, cfg) == [128]


# --- matching cap ----------------------------------------------------------------


def test_exact_retry_cannot_use_end_of_prompt_snapshot():
    a = toks(1, 256)
    _, rec = run(make_trace(("A", a, 1, None, None), ("R", a, 1, "A", "retry")))
    # The match stops at 255 tokens, before the only state at 256.
    assert (rec["R"].kv_hit, rec["R"].usable_prefix, rec["R"].retreat_tokens) == (255, 0, 255)
    assert POLICY.plan_prefill(res_req(a), 0, 255, SimConfig()) == [192]


# --- decode (R12, R13) -----------------------------------------------------------


@pytest.mark.parametrize(
    ("out", "roundtrip", "usable"),
    [(100, True, 256), (100, False, 192), (30, True, 192)],
)
def test_decode_snapshot(out, roundtrip, usable):
    a = toks(1, 200)
    b = a + toks(2, out) + toks(3, 30)
    _, rec = run(
        make_trace(("A", a, out, None, None), ("B", b, 1, "A", "append")),
        track_interval=128,
        output_roundtrip=roundtrip,
    )
    assert rec["B"].usable_prefix == usable


def test_decode_snapshot_needs_last_token_processed():
    # prompt 200 + out 57: decode processes up to 256 tokens, so 256 is recorded.
    assert POLICY.plan_finish(res_req(toks(1, 200), 57), SimConfig(track_interval=128)) == [256]
    # out 56: the 256th token is the final output and is never fed back.
    assert POLICY.plan_finish(res_req(toks(1, 200), 56), SimConfig(track_interval=128)) == []


def test_decode_snapshot_without_append_child_is_not_inserted():
    res, _ = run(make_trace(("A", toks(1, 200), 100, None, None)), track_interval=128)
    assert res.snapshots_written == 1


# --- eviction (R4, R14) ----------------------------------------------------------


def test_lru_with_one_slot():
    a = toks(1, 320)
    b = a + toks(2, 50)
    res, rec = run(
        make_trace(("A", a, 1, None, None), ("B", b, 1, "A", "append")),
        chunked_prefill_size=100,
        state_slots=1,
    )
    # A writes 64, 164, 264; each evicts the previous one. B resumes at 264, then its
    # 106-token extend writes 328 and evicts 264.
    assert (res.evictions, res.state_peak) == (3, 1)
    assert rec["B"].usable_prefix == 264
    assert rec["B"].snapshots_inserted == 1


def test_lru_refreshes_used_state_only():
    a, x, y = toks(1, 200), toks(2, 200), toks(3, 200)
    trace = make_trace(
        ("A", a, 1, None, None),
        ("X", x, 1, None, None),
        ("B", a + toks(4, 30), 1, "A", "append"),  # uses and refreshes A's state
        ("Y", y, 1, None, None),  # evicts X, the least recently used
        ("A2", a + toks(5, 30), 1, "A", "fork"),
        ("X2", x + toks(6, 30), 1, "X", "fork"),
    )
    res, rec = run(trace, state_slots=2)
    # Y evicts X; X2 then records the branching point 192 and evicts Y.
    assert res.evictions == 2
    assert rec["A2"].usable_prefix == 192
    # X's KV is still cached (KV capacity is not modeled) but its state is gone.
    assert (rec["X2"].kv_hit, rec["X2"].usable_prefix) == (192, 0)


def test_zero_slots_store_nothing():
    res, rec = run(
        make_trace(("A", toks(1, 200), 1, None, None), ("B", toks(1, 230), 1, "A", "append")),
        state_slots=0,
    )
    assert res.snapshots_written == 0
    assert rec["B"].usable_prefix == 0


def test_reinserting_existing_snapshot_refreshes_without_writing():
    tr = make_trace(*[(n, toks(t, 100), 1, None, None) for t, n in enumerate("abc", 1)])
    a, b, c = tr.requests
    sim = Simulation(tr, POLICY, SimConfig(state_slots=2))
    assert sim.store(a, 64, "a") == 1
    assert sim.store(b, 64, "b") == 1
    assert sim.store(a, 64, "a") == 0  # refresh: a is now the most recent
    assert sim.store(c, 64, "c") == 1  # evicts b
    assert (sim.count, sim.result.evictions) == (2, 1)
    assert sim.best_match(b, 99) is None
    assert sim.best_match(a, 99).pos == 64
    assert sim.store(a, 0, "a") == 0  # no position, nothing stored


# --- config, summary, API ----------------------------------------------------------


def test_grids():
    cfg = SimConfig(cache_chunk_size=64, page_size=16, track_interval=256)
    assert (cfg.checkpoint_grid, cfg.track_grid) == (64, 256)
    cfg = SimConfig(cache_chunk_size=128, page_size=128, track_interval=200)
    assert (cfg.checkpoint_grid, cfg.track_grid) == (128, 3200)


def test_summary_and_sweep():
    tr = synth.generate_trace(synth.preset_params("mixed", seed=3, sessions=4))
    rows = sweep(tr, [get_policy("sglang_current")], [None, 4], SimConfig())
    assert [r["config"]["state_slots"] for r in rows] == [None, 4]
    for r in rows:
        assert r["provenance"] == "simulated"
        assert 0 <= r["recompute_ratio"] <= 1
        assert r["recompute_tokens"] <= r["prompt_tokens"]
    assert simulate(tr, POLICY, SimConfig()).summary() == rows[0]


def test_simulation_is_deterministic():
    tr = synth.generate_trace(synth.preset_params("mixed", seed=5, sessions=3))
    a = simulate(tr, POLICY, SimConfig(state_slots=8))
    b = simulate(tr, POLICY, SimConfig(state_slots=8))
    assert a == b


def test_unknown_policy():
    with pytest.raises(ValueError, match="unknown policy"):
        get_policy("nope")


def test_cli_sim(tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    schema.dump_trace(synth.generate_trace(synth.preset_params("fork", seed=1, sessions=2)), path)
    assert cli.main(["sim", str(path), "--state-slots", "none", "--state-slots", "16"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["config"]["state_slots"] for r in rows] == [None, 16]
    assert all(r["policy"] == "sglang_current" for r in rows)

    assert cli.main(["sim", str(path), "--per-request"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["records"][0]["req_id"]

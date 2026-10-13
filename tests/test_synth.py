import pytest

from state_bench.traces import schema, synth
from state_bench.traces.synth import (
    AppendParams,
    ForkParams,
    LenDist,
    RetryParams,
    RewriteParams,
    TimeDist,
    TraceParams,
)


def fixed_params(*, fork=None, retry=None, rewrite=None, sessions=2, **over) -> TraceParams:
    """Fixed lengths so results can be checked by hand.

    Service time per turn is 0.5 + 20/40 = 1.0 s; tool wait is 2.0 s.
    """
    kw = dict(
        turns=LenDist("fixed", 4),
        system_tokens=LenDist("fixed", 100),
        user_tokens=LenDist("fixed", 50),
        output_tokens=LenDist("fixed", 20),
        tool_result_tokens=LenDist("fixed", 30),
        tool_wait=TimeDist("fixed", 2.0),
        nominal_ttft_s=0.5,
        nominal_decode_tps=40.0,
    )
    return TraceParams(
        seed=7,
        sessions=sessions,
        session_interarrival=TimeDist("fixed", 10.0),
        append=AppendParams(**{**kw, **over}),
        fork=fork or ForkParams(),
        retry=retry or RetryParams(),
        rewrite=rewrite or RewriteParams(),
    )


def s0(tr):
    return tr.requests_by_session()["s0000"]


# --- append -------------------------------------------------------------------


def test_hand_computed_lengths_and_times():
    tr = synth.generate_trace(fixed_params())
    reqs = s0(tr)
    assert [r.prompt_tokens for r in reqs] == [150, 200, 250, 300]
    assert [len(r.blocks) for r in reqs] == [3, 4, 4, 5]
    assert [r.t_arrival_s for r in reqs] == [0.0, 3.0, 6.0, 9.0]
    assert [r.tool_wait_s for r in reqs] == [2.0, 2.0, 2.0, None]
    assert [r.pattern for r in reqs] == [None, "append", "append", "append"]
    assert [s.t_start_s for s in tr.sessions] == [0.0, 10.0]


def test_segments_are_message_boundaries():
    last = s0(synth.generate_trace(fixed_params()))[-1]
    assert [s.kind for s in last.segments] == ["system", "user"] + ["assistant", "tool_result"] * 3
    assert [s.end_tok for s in last.segments] == [100, 150, 170, 200, 220, 250, 270, 300]


def test_append_reuses_all_parent_full_blocks():
    tr = synth.generate_trace(TraceParams(seed=3, sessions=3))
    idx = {r.req_id: r for r in tr.requests}
    for r in tr.requests:
        if r.pattern == "append":
            p = idx[r.parent_req_id]
            assert r.blocks[: len(p.full_blocks)] == p.full_blocks
            if p.last_block_tokens != schema.BLOCK_TOKENS:
                assert r.blocks[len(p.full_blocks)] != p.blocks[-1]
    assert schema.validate_trace(tr) == []


def test_shared_system_prompt_shares_prefix_across_sessions():
    tr = synth.generate_trace(fixed_params())
    a, b = (tr.requests_by_session()[s][0] for s in ("s0000", "s0001"))
    assert a.blocks[0] == b.blocks[0]
    assert a.blocks[1] != b.blocks[1]


def test_unshared_system_prompt():
    tr = synth.generate_trace(fixed_params(shared_system_prompt=False))
    a, b = (tr.requests_by_session()[s][0] for s in ("s0000", "s0001"))
    assert a.blocks[0] != b.blocks[0]


def test_context_limit_ends_session():
    tr = synth.generate_trace(fixed_params(max_prompt_tokens=260, turns=LenDist("fixed", 50)))
    reqs = s0(tr)
    assert [r.prompt_tokens for r in reqs] == [150, 200, 250]
    assert reqs[-1].tool_wait_s is None


# --- fork -------------------------------------------------------------------


def _fork_trace(share):
    fork = ForkParams(
        prob=1.0,
        share=share,
        task_tokens=LenDist("fixed", 40),
        turns=LenDist("fixed", 2),
        start_delay_s=0.05,
    )
    return synth.generate_trace(fixed_params(fork=fork, turns=LenDist("fixed", 2), sessions=1))


def test_fork_share_system_hand_computed():
    reqs = s0(_fork_trace("system"))
    main0, fork0, br1, main1 = reqs
    assert [r.branch_id for r in reqs] == [0, 1, 1, 0]
    assert [r.pattern for r in reqs] == [None, "fork", "append", "append"]
    assert [r.prompt_tokens for r in reqs] == [150, 140, 190, 200]
    # Branch starts at output end 1.0 + 0.05; next branch turn adds 1.0 service + 2.0 wait.
    assert fork0.t_arrival_s == pytest.approx(1.05)
    assert br1.t_arrival_s == pytest.approx(4.05)
    # The main chain waits for the branch to end (5.05), then 2.0 more.
    assert main0.tool_wait_s == pytest.approx(6.05)
    assert main1.t_arrival_s == pytest.approx(7.05)
    assert fork0.parent_req_id == main0.req_id and main1.parent_req_id == main0.req_id
    # Only the system prompt is shared: block 0 (system only) matches, block 1 differs.
    assert fork0.blocks[0] == main0.blocks[0] and fork0.blocks[1] != main0.blocks[1]
    assert [s.kind for s in fork0.segments] == ["system", "user"]


def test_fork_share_full_contains_parent():
    main0, fork0, *_ = s0(_fork_trace("full"))
    assert fork0.prompt_tokens == 150 + 40
    assert fork0.blocks[: len(main0.full_blocks)] == main0.full_blocks


def test_fork_width():
    fork = ForkParams(prob=1.0, width=LenDist("fixed", 3), turns=LenDist("fixed", 1))
    tr = synth.generate_trace(fixed_params(fork=fork, turns=LenDist("fixed", 2), sessions=1))
    forks = [r for r in s0(tr) if r.pattern == "fork"]
    assert len(forks) == 3 and len({r.branch_id for r in forks}) == 3
    assert schema.validate_trace(tr) == []


# --- retry -------------------------------------------------------------------


def test_retry_hand_computed():
    retry = RetryParams(prob=1.0, delay=TimeDist("fixed", 5.0))
    reqs = s0(
        synth.generate_trace(fixed_params(retry=retry, turns=LenDist("fixed", 2), sessions=1))
    )
    assert [r.pattern for r in reqs] == [None, "retry", "append", "retry"]
    assert [r.t_arrival_s for r in reqs] == [0.0, 5.0, 8.0, 13.0]
    assert reqs[1].blocks == reqs[0].blocks
    assert reqs[0].tool_wait_s is None and reqs[1].tool_wait_s == 2.0
    assert reqs[2].parent_req_id == reqs[1].req_id


# --- rewrite -------------------------------------------------------------------


def test_rewrite_hand_computed():
    rw = RewriteParams(prob=1.0, keep_messages=2, summary_tokens=LenDist("fixed", 64))
    reqs = s0(synth.generate_trace(fixed_params(rewrite=rw, turns=LenDist("fixed", 3), sessions=1)))
    assert [r.prompt_tokens for r in reqs] == [150, 214, 214]
    assert [r.pattern for r in reqs] == [None, "rewrite", "rewrite"]
    # System + first user message = 150 tokens -> 2 shared full blocks; summaries differ after.
    assert reqs[1].blocks[:2] == reqs[0].blocks[:2] == reqs[2].blocks[:2]
    assert reqs[1].blocks[2] != reqs[2].blocks[2]
    assert [s.kind for s in reqs[2].segments] == ["system", "user", "user"]


def test_overflow_compaction_keeps_session_going():
    rw = RewriteParams(on_overflow=True, summary_tokens=LenDist("fixed", 64))
    p = fixed_params(rewrite=rw, max_prompt_tokens=260, turns=LenDist("fixed", 6), sessions=1)
    reqs = s0(synth.generate_trace(p))
    assert [r.prompt_tokens for r in reqs] == [150, 200, 250, 214, 214, 214]
    assert [r.pattern for r in reqs] == [None, "append", "append", "rewrite", "rewrite", "rewrite"]


# --- whole trace -------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mixed_preset_valid_and_covers_all_patterns(seed):
    tr = synth.generate_trace(synth.preset_params("mixed", seed=seed, sessions=16))
    assert schema.validate_trace(tr) == []
    seen = {r.pattern for r in tr.requests}
    assert {"append", "fork", "retry"} <= seen
    for reqs in tr.requests_by_session().values():
        assert [r.t_arrival_s for r in reqs] == sorted(r.t_arrival_s for r in reqs)


def test_deterministic_and_seed_sensitive(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    schema.dump_trace(synth.generate_trace(synth.preset_params("mixed", seed=1)), p1)
    schema.dump_trace(synth.generate_trace(synth.preset_params("mixed", seed=1)), p2)
    assert p1.read_bytes() == p2.read_bytes()
    other = synth.generate_trace(synth.preset_params("mixed", seed=2))
    assert other.requests[0].blocks != schema.load_trace(p1).requests[0].blocks


def test_header_marks_params_as_assumption():
    tr = synth.generate_trace(TraceParams())
    assert tr.header.params["provenance"] == "assumption"
    assert tr.header.hash_scheme == synth.HASH_SCHEME


def test_prefix_hasher_incremental_and_truncate():
    data = synth.message_bytes(b"x", 200)
    inc = synth.PrefixHasher()
    inc.extend(data[: 37 * 4])
    inc.extend(data[37 * 4 :])
    one = synth.PrefixHasher()
    one.extend(data)
    assert inc.blocks() == one.blocks()
    assert len(one.blocks()) == 4
    short = synth.PrefixHasher()
    short.extend(data[: 130 * 4])
    assert one.truncated(130).blocks() == short.blocks()


def test_unknown_preset():
    with pytest.raises(ValueError):
        synth.preset_params("nope")


def test_cli_roundtrip(tmp_path, capsys):
    from state_bench import cli

    out = tmp_path / "t.jsonl"
    assert (
        cli.main(["synth", "--preset", "mixed", "--sessions", "4", "--seed", "5", "-o", str(out)])
        == 0
    )
    assert cli.main(["validate", str(out)]) == 0
    assert cli.main(["stats", str(out)]) == 0
    assert "parent_reusable_token_ratio" in capsys.readouterr().out


def test_cli_schema_prints_json_schema(capsys):
    import json

    from state_bench import cli

    assert cli.main(["schema"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["title"] == "state-bench trace record"

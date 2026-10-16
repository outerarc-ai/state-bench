"""Tests for replay materialization: deterministic tokens that keep prefix structure."""

from __future__ import annotations

import json

import pytest
from sim_helpers import make_trace, toks

from state_bench import cli
from state_bench.replay import (
    MaterializedRequest,
    MaterializedTrace,
    check_prefix_fidelity,
    materialize,
    shared_block_tokens,
    shared_materialized_prefix,
    write_replay,
)
from state_bench.traces import schema, synth
from state_bench.traces.schema import Request, Session, Trace, TraceHeader

VOCAB = 4096


# ---------------------------------------------------------------------------
# Token derivation
# ---------------------------------------------------------------------------


def test_block_token_ids_deterministic_and_in_range():
    from state_bench.replay.materialize import _block_token_ids

    a = _block_token_ids("0123456789abcdef", 8, VOCAB)
    b = _block_token_ids("0123456789abcdef", 8, VOCAB)
    assert a == b
    assert len(a) == 8
    assert all(0 <= t < VOCAB for t in a)
    assert _block_token_ids("fedcba9876543210", 8, VOCAB) != a


def test_materialize_length_for_all_block_shapes():
    for n in (1, 63, 64, 65, 128, 200):
        trace = make_trace(("r", toks(1, n), 3, None, None))
        assert len(materialize(trace, vocab_size=VOCAB).requests[0].prompt) == n


def test_materialize_is_deterministic():
    trace = make_trace(
        ("a", toks(1, 100), 3, None, None),
        ("b", toks(1, 100) + toks(2, 40), 3, "a", "append"),
    )
    assert materialize(trace, vocab_size=VOCAB) == materialize(trace, vocab_size=VOCAB)


def test_materialize_rejects_bad_vocab():
    trace = make_trace(("a", toks(1, 64), 1, None, None))
    with pytest.raises(ValueError, match="vocab_size"):
        materialize(trace, vocab_size=0)


# ---------------------------------------------------------------------------
# Prefix fidelity
# ---------------------------------------------------------------------------


def _lineage_trace() -> Trace:
    base = toks(1, 100) + toks(2, 50)  # 150 tokens: 2 full blocks + a 22-token tail
    ext = base + toks(3, 80)  # 230
    fork = base + toks(9, 120)  # 270, diverges after `base`
    return make_trace(
        ("a", base, 5, None, None),
        ("b", ext, 5, "a", "append"),
        ("c", ext, 3, "b", "retry"),
        ("d", fork, 4, "b", "fork"),
        ("e", toks(7, 60), 2, None, None),
        times=[0, 1, 2, 3, 4],
    )


def test_lineage_sharing_materializes_exactly():
    trace = _lineage_trace()
    assert schema.validate_trace(trace) == []
    mat = materialize(trace, vocab_size=VOCAB)
    by = mat.by_req_id()

    # append shares the parent's whole prompt, its mid-block tail included
    assert by["b"].prompt[: len(by["a"].prompt)] == by["a"].prompt
    assert len(by["a"].prompt) == 150
    # retry is identical, tail included
    assert by["c"].prompt == by["b"].prompt
    # fork shares only the common full-block prefix
    assert (
        shared_materialized_prefix(by["d"].prompt, by["b"].prompt)
        == shared_block_tokens(trace.requests[3], trace.requests[1])
        == 128
    )
    # an unrelated root shares nothing
    assert shared_materialized_prefix(by["e"].prompt, by["a"].prompt) == 0


def test_check_prefix_fidelity_passes_on_lineage_trace():
    trace = _lineage_trace()
    assert check_prefix_fidelity(trace, materialize(trace, vocab_size=VOCAB)) == []


def test_shared_block_tokens_counts_shared_tail():
    trace = _lineage_trace()
    b, c = trace.requests[1], trace.requests[2]
    assert shared_block_tokens(b, c) == b.prompt_tokens == 230


def test_check_prefix_fidelity_passes_across_shared_system():
    trace = synth.generate_trace(synth.preset_params("mixed", sessions=3, seed=1))
    mat = materialize(trace, vocab_size=VOCAB)
    assert check_prefix_fidelity(trace, mat) == []
    # Different sessions share the system prompt, which is more than lineage covers.
    first = [r for r in trace.requests if r.prompt_tokens > 64]
    assert (
        shared_materialized_prefix(
            mat.by_req_id()[first[0].req_id].prompt, mat.by_req_id()[first[1].req_id].prompt
        )
        > 64
    )


def test_check_prefix_fidelity_flags_tampered_tokens():
    trace = _lineage_trace()
    mat = materialize(trace, vocab_size=VOCAB)
    bad = MaterializedRequest(
        req_id=mat.requests[0].req_id,
        session_id=mat.requests[0].session_id,
        prompt=tuple([0] * len(mat.requests[0].prompt)),
        output_len=mat.requests[0].output_len,
        timestamp_ms=mat.requests[0].timestamp_ms,
    )
    tampered = MaterializedTrace(vocab_size=mat.vocab_size, requests=(bad, *mat.requests[1:]))
    assert check_prefix_fidelity(trace, tampered) != []


def test_check_prefix_fidelity_flags_length_mismatch():
    trace = _lineage_trace()
    mat = materialize(trace, vocab_size=VOCAB)
    short = MaterializedRequest(
        req_id=mat.requests[0].req_id,
        session_id=mat.requests[0].session_id,
        prompt=mat.requests[0].prompt[:-1],
        output_len=0,
        timestamp_ms=0.0,
    )
    tampered = MaterializedTrace(vocab_size=mat.vocab_size, requests=(short, *mat.requests[1:]))
    errs = check_prefix_fidelity(trace, tampered)
    assert any("tokens, expected" in e for e in errs)


# ---------------------------------------------------------------------------
# Timeline and output
# ---------------------------------------------------------------------------


def test_global_arrival_timeline_uses_session_starts():
    def mk(rid: str, sid: str, arr: float) -> Request:
        return Request(
            req_id=rid,
            session_id=sid,
            parent_req_id=None,
            branch_id=0,
            t_arrival_s=arr,
            prompt_tokens=64,
            blocks=("0123456789abcdef",),
            segments=None,
            output_tokens=1,
            tool_wait_s=None,
            pattern=None,
        )

    trace = Trace(
        TraceHeader(),
        [
            Session("s0", 0.0, "m", "test", "Apache-2.0"),
            Session("s1", 5.0, "m", "test", "Apache-2.0"),
        ],
        [mk("late", "s1", 1.0), mk("early", "s0", 2.0)],
    )
    mat = materialize(trace, vocab_size=VOCAB)
    assert [r.req_id for r in mat.requests] == ["early", "late"]
    assert [r.timestamp_ms for r in mat.requests] == [2000.0, 6000.0]


def test_dump_autobench_fields_and_omitted_output_len(tmp_path):
    trace = make_trace(
        ("a", toks(1, 70), 0, None, None),
        ("b", toks(1, 70) + toks(2, 70), 9, "a", "append"),
    )
    out = tmp_path / "replay.jsonl"
    write_replay(trace, out, vocab_size=VOCAB)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["prompt_len"] for r in recs] == [70, 140]
    assert all(len(r["prompt"]) == r["prompt_len"] for r in recs)
    assert "output_len" not in recs[0]  # trace recorded none -> consumer default applies
    assert recs[1]["output_len"] == 9
    assert recs[0]["timestamp"] <= recs[1]["timestamp"]


def test_write_replay_sidecar_records_provenance(tmp_path):
    trace = _lineage_trace()
    out = tmp_path / "replay.jsonl"
    write_replay(trace, out, vocab_size=VOCAB)
    meta = json.loads((tmp_path / "replay.jsonl.meta.json").read_text())
    assert meta["vocab_size"] == VOCAB
    assert meta["provenance"]["token_ids"] == "assumption"
    assert meta["materialized"]["requests"] == len(trace.requests)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_materialize(tmp_path, capsys):
    trace = _lineage_trace()
    src = tmp_path / "t.jsonl"
    schema.dump_trace(trace, src)
    out = tmp_path / "r.jsonl"
    assert cli.main(["materialize", str(src), "-o", str(out), "--vocab-size", str(VOCAB)]) == 0
    assert out.exists()
    assert (tmp_path / "r.jsonl.meta.json").exists()
    assert "provenance: assumption" in capsys.readouterr().err


def test_cli_materialize_no_check(tmp_path):
    trace = _lineage_trace()
    src = tmp_path / "t.jsonl"
    schema.dump_trace(trace, src)
    out = tmp_path / "r.jsonl"
    assert (
        cli.main(["materialize", str(src), "-o", str(out), "--vocab-size", "8", "--no-check"]) == 0
    )

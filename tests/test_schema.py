import dataclasses
import json

import pytest

from state_bench.traces import schema
from state_bench.traces.schema import Request, Segment, Session, Trace, TraceHeader

H = "0123456789abcdef"


def _h(i: int) -> str:
    return f"{i:016x}"


def _req(req_id, prompt, blocks, *, parent=None, pattern=None, t=0.0, segs="auto", sid="s0"):
    if segs == "auto":
        segs = (Segment("system", 0, prompt),)
    return Request(
        req_id=req_id,
        session_id=sid,
        parent_req_id=parent,
        branch_id=0,
        t_arrival_s=t,
        prompt_tokens=prompt,
        blocks=tuple(blocks),
        segments=segs,
        output_tokens=10,
        tool_wait_s=None,
        pattern=pattern,
    )


def _trace(*reqs):
    return Trace(TraceHeader(), [Session("s0", 0.0, "qwen3.5", "test", "Apache-2.0")], list(reqs))


def test_last_block_and_full_blocks():
    r = _req("a", 130, [_h(1), _h(2), _h(3)])
    assert r.last_block_tokens == 2
    assert r.full_blocks == (_h(1), _h(2))
    r2 = _req("b", 128, [_h(1), _h(2)])
    assert r2.last_block_tokens == 64
    assert r2.full_blocks == (_h(1), _h(2))


def test_valid_append_chain_passes():
    a = _req("a", 100, [_h(1), _h(2)])
    b = _req("b", 200, [_h(1), _h(3), _h(4), _h(5)], parent="a", pattern="append", t=1.0)
    # b shares a's only full block and is longer: a valid append
    assert schema.validate_trace(_trace(a, b)) == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda b: dataclasses.replace(b, blocks=b.blocks[:-1]), "blocks, got"),
        (lambda b: dataclasses.replace(b, blocks=(_h(9), *b.blocks[1:])), "append must extend"),
        (lambda b: dataclasses.replace(b, parent_req_id="zzz"), "is missing"),
        (lambda b: dataclasses.replace(b, pattern=None), "both be set"),
        (lambda b: dataclasses.replace(b, t_arrival_s=-0.0, session_id="s0"), None),
        (
            lambda b: dataclasses.replace(
                b, segments=(Segment("system", 0, 50), Segment("user", 60, 200))
            ),
            "not contiguous",
        ),
        (
            lambda b: dataclasses.replace(b, segments=(Segment("system", 0, 150),)),
            "segments end at",
        ),
        (lambda b: dataclasses.replace(b, session_id="nope"), "not declared"),
    ],
)
def test_semantic_errors(mutate, needle):
    a = _req("a", 100, [_h(1), _h(2)], t=0.5)
    b = _req("b", 200, [_h(1), _h(3), _h(4), _h(5)], parent="a", pattern="append", t=1.0)
    errs = schema.validate_trace(_trace(a, mutate(b)))
    if needle is None:  # arrives before its parent
        assert any("earlier than its parent" in e for e in errs)
    else:
        assert any(needle in e for e in errs), errs


def test_retry_must_match_parent():
    a = _req("a", 100, [_h(1), _h(2)])
    ok = _req("b", 100, [_h(1), _h(2)], parent="a", pattern="retry", t=1)
    bad = _req("c", 100, [_h(1), _h(7)], parent="a", pattern="retry", t=1)
    assert schema.validate_trace(_trace(a, ok)) == []
    assert any("retry must" in e for e in schema.validate_trace(_trace(a, bad)))


def test_missing_segments_allowed_but_empty_list_rejected():
    a = _req("a", 100, [_h(1), _h(2)], segs=None)
    assert schema.validate_trace(_trace(a)) == []
    b = _req("a", 100, [_h(1), _h(2)], segs=())
    assert any("use null" in e for e in schema.validate_trace(_trace(b)))


def test_roundtrip_and_json_schema(tmp_path):
    a = _req("a", 100, [_h(1), _h(2)])
    b = _req("b", 200, [_h(1), _h(3), _h(4), _h(5)], parent="a", pattern="append", t=1.0)
    tr = _trace(a, b)
    p = tmp_path / "t.jsonl"
    schema.dump_trace(tr, p)
    back = schema.load_trace(p)
    assert back.requests == tr.requests
    assert back.sessions == tr.sessions


def test_json_schema_rejects_bad_hash(tmp_path):
    a = _req("a", 100, ["NOT-A-HASH", _h(2)])
    p = tmp_path / "t.jsonl"
    schema.dump_trace(_trace(a), p)
    with pytest.raises(schema.TraceError, match="line 3"):
        schema.load_trace(p)


def test_header_must_be_first(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"type": "session"}) + "\n")
    with pytest.raises(schema.TraceError):
        schema.parse_records([json.loads(p.read_text())])


def test_global_arrivals_orders_across_sessions():
    s0 = Session("s0", 0.0, "m", "t", "x")
    s1 = Session("s1", 5.0, "m", "t", "x")
    r0 = _req("a", 64, [_h(1)], t=10.0)
    r1 = _req("b", 64, [_h(1)], t=1.0, sid="s1")
    tr = Trace(TraceHeader(), [s0, s1], [r0, r1])
    assert [r.req_id for _, r in tr.global_arrivals()] == ["b", "a"]

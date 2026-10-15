import json

import pytest

from state_bench import cli
from state_bench.traces import schema, weka


def req(t, n_in, ids, out=10, think=0.0, **extra):
    return {
        "t": t,
        "type": "n",
        "in": n_in,
        "out": out,
        "hash_ids": ids,
        "think_time": think,
        **extra,
    }


def trace(tid="tr1", requests=(), **extra):
    return {
        "id": tid,
        "models": ["model-a"],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 100,
        "system_tokens": 50,
        "requests": list(requests),
        **extra,
    }


def convert(t, raw=None):
    return weka.convert_trace(t, "f.json", "corpus", "Apache-2.0", raw_out=raw)


def by_id(reqs):
    return {r.req_id: r for r in reqs}


def test_append_chain_tail_blocks_and_waits():
    t = trace(
        requests=[
            req(0.0, 130, [1, 2], think=0.0, stop="tool_use"),
            req(5.0, 200, [1, 2, 3], think=1.5),
            req(9.0, 256, [1, 2, 3, 4], think=2.0),
        ]
    )
    raw = {}
    session, reqs, prefix = convert(t, raw)
    assert prefix == 150
    assert session.session_id == "tr1" and session.source == "corpus:f.json"
    a, b, c = reqs
    assert [r.req_id for r in reqs] == ["tr1-m-0000", "tr1-m-0001", "tr1-m-0002"]
    assert a.parent_req_id is None and a.pattern is None
    assert (b.parent_req_id, b.pattern) == (a.req_id, "append")
    assert (c.parent_req_id, c.pattern) == (b.req_id, "append")
    # 130 tokens: 2 full blocks and a partial one with a request-unique hash.
    assert len(a.blocks) == 3 and a.blocks[:2] == b.blocks[:2]
    assert a.blocks[2] == weka.tail_hash("tr1", a.req_id) != b.blocks[2]
    assert b.blocks[2] == weka.block_hash("tr1", 3)
    assert len(c.blocks) == 4  # 256 tokens, no partial block
    assert (a.tool_wait_s, b.tool_wait_s, c.tool_wait_s) == (1.5, 2.0, None)
    assert raw[a.req_id]["stop"] == "tool_use"
    assert not schema.validate_trace(schema.Trace(schema.TraceHeader(), [session], reqs))


def test_rewrite_retry_root_and_latest_parent_on_tie():
    t = trace(
        requests=[
            req(0.0, 192, [1, 2, 3]),
            req(1.0, 300, [1, 2, 3, 4]),
            req(2.0, 140, [1, 2]),  # keeps 2 of 4 blocks: rewrite of the latest sharer
            req(3.0, 140, [1, 2]),  # same blocks and length as the previous: retry
            req(4.0, 70, [9]),  # shares nothing: a new root
        ]
    )
    _, reqs, _ = convert(t)
    r = by_id(reqs)
    assert (r["tr1-m-0002"].pattern, r["tr1-m-0002"].parent_req_id) == ("rewrite", "tr1-m-0001")
    assert (r["tr1-m-0003"].pattern, r["tr1-m-0003"].parent_req_id) == ("retry", "tr1-m-0002")
    assert r["tr1-m-0004"].parent_req_id is None and r["tr1-m-0004"].pattern is None
    # A rewrite child supplies its parent's wait.
    assert r["tr1-m-0001"].tool_wait_s == 0.0


def test_subagents_fork_nest_and_get_branch_ids():
    sub_inner = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "agent_002",
        "requests": [req(0.0, 64, [50]), req(2.0, 128, [50, 51], think=0.5)],
    }
    sub = {
        "t": 3.0,
        "type": "subagent",
        "agent_id": "agent_001",
        "requests": [req(0.0, 100, [20], think=7.0), sub_inner, req(4.0, 150, [20, 21], think=1.0)],
    }
    t = trace(
        requests=[
            req(0.0, 64, [1]),
            req(2.0, 128, [1, 2], think=1.0),
            sub,
            req(10.0, 200, [1, 2, 3]),
        ]
    )
    session, reqs, _ = convert(t)
    r = by_id(reqs)
    first_sub = r["tr1-m-agent_001-0000"]
    assert (first_sub.pattern, first_sub.parent_req_id) == ("fork", "tr1-m-0001")
    assert first_sub.t_arrival_s == 3.0
    nested = r["tr1-m-agent_001-agent_002-0000"]
    assert (nested.pattern, nested.parent_req_id) == ("fork", "tr1-m-agent_001-0000")
    assert nested.t_arrival_s == 4.0
    assert r["tr1-m-agent_001-agent_002-0001"].t_arrival_s == 6.0
    assert r["tr1-m-agent_001-0001"].pattern == "append"
    assert {x.branch_id for x in reqs if "agent" not in x.req_id} == {0}
    assert first_sub.branch_id == 1 and nested.branch_id == 2
    # A fork child does not set the spawner's wait; the main agent's own child does.
    assert r["tr1-m-0001"].tool_wait_s == 0.0
    assert [x.t_arrival_s for x in reqs] == sorted(x.t_arrival_s for x in reqs)
    assert not schema.validate_trace(schema.Trace(schema.TraceHeader(), [session], reqs))


def test_fork_without_earlier_spawner_request_is_root():
    sub = {"t": 0.0, "type": "subagent", "agent_id": "a", "requests": [req(0.0, 64, [5])]}
    t = trace(requests=[sub, req(1.0, 64, [1])])
    _, reqs, _ = convert(t)
    assert all(x.parent_req_id is None for x in reqs)


@pytest.mark.parametrize(
    "bad, message",
    [
        (trace(requests=[req(0.0, 0, [])]), "input tokens"),
        (trace(requests=[req(0.0, 100, [1, 2])]), "full blocks"),
        (trace(requests=[req(0.0, 64, [1])], block_size=16), "block_size"),
        (
            trace(
                requests=[
                    {"t": 0, "type": "subagent", "agent_id": "x", "requests": []},
                    {"t": 1, "type": "subagent", "agent_id": "x", "requests": []},
                ]
            ),
            "duplicate agent",
        ),
    ],
)
def test_rejects_malformed_traces(bad, message):
    with pytest.raises(schema.TraceError, match=message):
        convert(bad)


def write_corpus(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()
    for i in (2, 1):
        t = trace(f"trace_{i}", [req(0.0, 64, [1]), req(1.0, 130, [1, 2], think=0.5)])
        (d / f"trace_{i}.json").write_text(json.dumps(t))
    (d / "notes.txt").write_text("ignored")
    jl = tmp_path / "more.jsonl"
    jl.write_text(json.dumps(trace("trace_3", [req(0.0, 64, [7])])) + "\n\n")
    return d, jl


def test_import_and_streaming_writer_agree(tmp_path):
    d, jl = write_corpus(tmp_path)
    kw = dict(source="corpus@abc", license_="Apache-2.0", session_interval_s=10.0)
    t = weka.import_weka([d, jl], **kw)
    assert [s.session_id for s in t.sessions] == ["trace_1", "trace_2", "trace_3"]
    assert [s.t_start_s for s in t.sessions] == [0.0, 10.0, 20.0]
    assert t.sessions[2].source == "corpus@abc:more.jsonl:1"
    assert t.header.hash_scheme == "weka-local-v1"
    assert t.header.params["prefix_tokens"] == {"trace_1": 150, "trace_2": 150, "trace_3": 150}
    # Same local IDs in different traces never share a hash.
    assert t.requests[0].blocks[0] != t.requests[2].blocks[0]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    schema.dump_trace(t, a)
    assert weka.write_import([d, jl], b, **kw) == (3, 5)
    assert a.read_bytes() == b.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
    loaded = schema.load_trace(b, check=True)
    assert len(loaded.requests) == 5
    assert len(weka.import_weka([d], limit=1, **kw).sessions) == 1


def test_import_rejects_duplicates_and_bad_interval(tmp_path):
    d, _ = write_corpus(tmp_path)
    with pytest.raises(schema.TraceError, match="duplicate trace id"):
        weka.import_weka([d, d], source="s", license_="l")
    with pytest.raises(ValueError):
        weka.import_weka([d], source="s", license_="l", session_interval_s=-1)


def test_write_import_validates_each_session(tmp_path, monkeypatch):
    d, _ = write_corpus(tmp_path)
    monkeypatch.setattr(weka, "validate_trace", lambda _t: ["broken"])
    with pytest.raises(schema.TraceError, match="broken"):
        weka.write_import([d], tmp_path / "o.jsonl", source="s", license_="l")
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_import(tmp_path, capsys):
    d, jl = write_corpus(tmp_path)
    out = tmp_path / "o.jsonl"
    args = ["import", "weka", str(d), str(jl), "-o", str(out), "--source", "c", "--license", "L"]
    assert cli.main(args) == 0
    assert "3 sessions, 5 requests" in capsys.readouterr().err
    assert cli.main(["validate", str(out)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(trace("x", [req(0.0, 100, [1, 2])])))
    args = ["import", "weka", str(bad), "-o", str(out), "--source", "c", "--license", "L"]
    assert cli.main(args) == 1
    assert "import failed" in capsys.readouterr().err

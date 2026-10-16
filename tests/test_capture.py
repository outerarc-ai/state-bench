"""Tests for the capture package: hashing, lineage inference, recorder, proxy."""

from __future__ import annotations

import importlib.util
import json
import random
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from state_bench.capture import (
    CapturedRequest,
    OpenAIRecorder,
    ProxyServer,
    TokenBlockHasher,
    TraceSink,
)
from state_bench.traces.schema import Segment, TraceError, load_trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tok_ids(seed: int, n: int) -> list[int]:
    return [random.Random(seed).randrange(1, 1_000_000) for _ in range(n)]


def blocks_of(ids: list[int]) -> tuple[str, ...]:
    h = TokenBlockHasher()
    h.extend(ids)
    return h.blocks()


def make_sink() -> TraceSink:
    s = TraceSink(model_family="m", source="test", license_="Apache-2.0")
    s.start_session("c0000", 0.0)
    return s


def cap(
    sink: TraceSink, ids: list[int], t_arr: float, t_done: float, ann: dict | None = None
) -> None:
    sink.request(
        CapturedRequest(
            session_id="c0000",
            t_arrival_s=t_arr,
            t_done_s=t_done,
            prompt_tokens=len(ids),
            blocks=blocks_of(ids),
            segments=None,
            output_tokens=0,
            annotations=ann or {},
        )
    )


# ---------------------------------------------------------------------------
# TokenBlockHasher
# ---------------------------------------------------------------------------


def test_hasher_block_count_and_determinism():
    ids = tok_ids(1, 200)
    h1, h2 = TokenBlockHasher(), TokenBlockHasher()
    h1.extend(ids)
    h2.extend(ids)
    assert h1.n_tokens == 200
    assert len(h1.blocks()) == 4  # 3 full + 8-token tail
    assert h1.blocks() == h2.blocks()


def test_hasher_tail_distinguishes_from_full_block():
    a, b = tok_ids(1, 63), tok_ids(1, 64)
    assert blocks_of(a)[0] != blocks_of(b)[0]  # same ids, different block length


def test_hasher_chain_and_prefix_sharing():
    base = tok_ids(1, 200)
    ext = base + tok_ids(2, 150)
    other = base[:128] + tok_ids(3, 300)
    from state_bench.capture.base import shared_full_block_hashes

    assert shared_full_block_hashes(blocks_of(ext), blocks_of(base)) == 3
    assert shared_full_block_hashes(blocks_of(ext), blocks_of(other)) == 2
    # Changing one token invalidates that block and everything after it.
    mutated = base + tok_ids(2, 150)
    mutated[100] += 1
    shared = shared_full_block_hashes(blocks_of(mutated), blocks_of(ext))
    assert shared == 1


# ---------------------------------------------------------------------------
# TraceSink lineage and pattern inference
# ---------------------------------------------------------------------------


def test_sink_full_scenario(tmp_path):
    sink = make_sink()
    base = tok_ids(1, 200)
    r1 = base
    r2 = base + tok_ids(2, 150)
    r4 = base[:128] + tok_ids(3, 300)
    r5 = base[:128] + tok_ids(4, 100)
    r6 = tok_ids(5, 300)

    cap(sink, r1, 0.0, 1.0, ann={"usage": {"prompt_tokens": 200}})
    cap(sink, r2, 3.0, 4.0)
    cap(sink, r2, 6.0, 7.0)  # retry of r2
    cap(sink, r4, 10.0, 11.0)  # fork off the retry
    cap(sink, r5, 13.0, 14.0)  # rewrite (shorter, proper prefix)
    cap(sink, r6, 20.0, 21.0)  # new conversation (root)

    out = tmp_path / "cap.jsonl"
    trace = sink.finish(out)
    reqs = {r.req_id: r for r in trace.requests}

    assert reqs["c0000-r0000"].pattern is None
    assert reqs["c0000-r0001"].pattern == "append"
    assert reqs["c0000-r0001"].parent_req_id == "c0000-r0000"
    assert reqs["c0000-r0002"].pattern == "retry"
    assert reqs["c0000-r0002"].parent_req_id == "c0000-r0001"
    assert reqs["c0000-r0003"].pattern == "fork"
    assert reqs["c0000-r0003"].parent_req_id == "c0000-r0002"
    assert reqs["c0000-r0003"].branch_id == 1
    assert reqs["c0000-r0004"].pattern == "rewrite"
    assert reqs["c0000-r0004"].parent_req_id == "c0000-r0003"
    assert reqs["c0000-r0004"].branch_id == 1
    assert reqs["c0000-r0005"].pattern is None

    # tool_wait: first append/retry child's arrival minus parent completion.
    assert reqs["c0000-r0000"].tool_wait_s == 2.0
    assert reqs["c0000-r0001"].tool_wait_s == 2.0
    assert reqs["c0000-r0002"].tool_wait_s is None
    assert reqs["c0000-r0003"].tool_wait_s is None

    # The written file passes full structural + semantic validation.
    loaded = load_trace(out, check=True)
    assert len(loaded.requests) == 6

    sidecar = json.loads((tmp_path / "cap.jsonl.annotations.json").read_text())
    assert sidecar["c0000-r0000"]["usage"]["prompt_tokens"] == 200
    assert "c0000-r0001" not in sidecar


def test_sink_rejects_bad_submission():
    sink = make_sink()
    with pytest.raises(TraceError, match="not declared"):
        sink.request(CapturedRequest("nope", 0.0, 1.0, 64, blocks_of(tok_ids(1, 64)), None, 0))
    with pytest.raises(TraceError, match="arrives before"):
        cap(sink, tok_ids(1, 128), 5.0, 6.0)
        extended = tok_ids(1, 128)
        extended.append(7)
        cap(sink, extended, 4.0, 5.0)


def test_sink_rejects_invalid_segments_on_finish(tmp_path):
    sink = make_sink()
    sink.request(
        CapturedRequest(
            session_id="c0000",
            t_arrival_s=0.0,
            t_done_s=1.0,
            prompt_tokens=200,
            blocks=blocks_of(tok_ids(1, 200)),
            segments=(Segment("user", 0, 100),),  # does not cover the input
            output_tokens=0,
        )
    )
    with pytest.raises(TraceError, match="failed validation"):
        sink.finish(tmp_path / "bad.jsonl")


# ---------------------------------------------------------------------------
# OpenAIRecorder
# ---------------------------------------------------------------------------

MESSAGES = [
    {"role": "system", "content": "abc"},
    {"role": "user", "content": "def"},
]


class FakeTokenizer:
    """Deterministic stand-in: 3 header tokens + 1 token per content char per message."""

    def _msg_tokens(self, i: int, m: dict) -> list[int]:
        role = m.get("role", "")
        content = m.get("content") or ""
        return [1000 + i, 5000 + len(role), 7000 + len(content)] + [
            700 + j for j in range(len(content))
        ]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        out: list[int] = []
        for i, m in enumerate(messages):
            out += self._msg_tokens(i, m)
        if add_generation_prompt:
            out += [9001, 9002]
        return out


def recorder() -> OpenAIRecorder:
    sink = TraceSink(model_family="fake", source="test", license_="Apache-2.0")
    return OpenAIRecorder(sink, FakeTokenizer(), model_family="fake")


def test_recorder_segments_and_usage():
    rec = recorder()
    rec.record(MESSAGES, None, 0.0, 1.0, {"prompt_tokens": 14, "completion_tokens": 5})
    captured = rec.sink._order[0]
    assert captured.prompt_tokens == 14
    assert captured.output_tokens == 5
    assert captured.segments == (
        Segment("system", 0, 6),
        Segment("user", 6, 14),
    )
    assert captured.tool_wait_s is None


def test_recorder_unknown_role_gives_null_segments():
    rec = recorder()
    rec.record(
        [{"role": "system", "content": "a"}, {"role": "function", "content": "b"}],
        None,
        0.0,
        1.0,
        None,
    )
    assert rec.sink._order[0].segments is None


def test_recorder_tool_role_maps_to_tool_result():
    rec = recorder()
    rec.record([{"role": "tool", "content": "ok"}], None, 0.0, 1.0, None)
    # 5 message tokens + 2 generation-prompt tokens, tail absorbed into the message.
    assert rec.sink._order[0].segments == (Segment("tool_result", 0, 7),)


def test_recorder_usage_mismatch_annotated():
    rec = recorder()
    rec.record(MESSAGES, None, 0.0, 1.0, {"prompt_tokens": 999, "completion_tokens": 1})
    req_id, ann = rec.sink._annotations[0]
    assert req_id == "c0000-r0000"
    assert ann["prompt_tokens_local_vs_server"] == [14, 999]
    assert ann["usage"]["prompt_tokens"] == 999


def test_recorder_skips_unrenderable_prompt():
    class BoomTokenizer:
        def apply_chat_template(self, *a, **kw):
            raise TypeError("nope")

    rec = OpenAIRecorder(recorder().sink, BoomTokenizer(), model_family="fake")
    rec.record(MESSAGES, None, 0.0, 1.0, None)
    assert rec.sink._order == []


# ---------------------------------------------------------------------------
# Proxy integration (fake upstream, localhost only)
# ---------------------------------------------------------------------------

UPSTREAM_STATE: dict = {"n": 0, "last": None}


class _UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        payload = json.dumps({"data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(body)
        UPSTREAM_STATE["n"] += 1
        UPSTREAM_STATE["last"] = req
        if req.get("messages") == [{"role": "user", "content": "__error__"}]:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "bad"}')
            return
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"h"}}]}\n\n')
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"i"}}]}\n\n')
            self.wfile.write(
                b'data: {"choices":[],"usage":{"prompt_tokens":14,"completion_tokens":2}}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 14, "completion_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)


def _proxy_setup():
    up = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    up_url = f"http://127.0.0.1:{up.server_address[1]}"
    sink = TraceSink(model_family="fake-model", source="proxy:openai", license_="Apache-2.0")
    rec = OpenAIRecorder(sink, FakeTokenizer(), model_family="fake-model")
    proxy = ProxyServer(up_url, rec)
    proxy.start()
    return up, proxy, sink


def _post(url: str, body: dict) -> tuple[int, bytes]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_proxy_nonstream_roundtrip(tmp_path):
    up, proxy, sink = _proxy_setup()
    try:
        status, payload = _post(
            f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
            {"model": "fake", "messages": MESSAGES},
        )
        assert status == 200
        assert json.loads(payload)["usage"]["completion_tokens"] == 3
        # GET passthrough (e.g. model probing).
        with urllib.request.urlopen(f"http://127.0.0.1:{proxy.port}/v1/models") as r:
            assert json.loads(r.read())["data"][0]["id"] == "fake-model"
    finally:
        proxy.stop()
        up.shutdown()

    out = tmp_path / "cap.jsonl"
    trace = sink.finish(out)
    assert len(trace.requests) == 1
    r0 = trace.requests[0]
    assert r0.pattern is None
    assert r0.prompt_tokens == 14
    assert r0.output_tokens == 3
    assert len(r0.blocks) == 1
    load_trace(out, check=True)
    sidecar = json.loads((tmp_path / "cap.jsonl.annotations.json").read_text())
    assert sidecar[r0.req_id]["usage"]["completion_tokens"] == 3


def test_proxy_stream_usage_and_retry_lineage(tmp_path):
    up, proxy, sink = _proxy_setup()
    try:
        url = f"http://127.0.0.1:{proxy.port}/v1/chat/completions"
        status, payload = _post(url, {"model": "fake", "messages": MESSAGES, "stream": True})
        assert status == 200
        assert b"data: [DONE]" in payload
        # The proxy injects include_usage so output length is measured.
        assert UPSTREAM_STATE["last"]["stream_options"] == {"include_usage": True}
        # Same prompt again -> retry lineage.
        status, _ = _post(url, {"model": "fake", "messages": MESSAGES})
        assert status == 200
    finally:
        proxy.stop()
        up.shutdown()

    out = tmp_path / "cap.jsonl"
    trace = sink.finish(out)
    assert [r.pattern for r in trace.requests] == [None, "retry"]
    assert trace.requests[0].output_tokens == 2  # from the final stream chunk
    assert trace.requests[1].output_tokens == 3


def test_proxy_upstream_error_not_recorded():
    up, proxy, sink = _proxy_setup()
    try:
        status, payload = _post(
            f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
            {"model": "fake", "messages": [{"role": "user", "content": "__error__"}]},
        )
        assert status == 400
        assert b"bad" in payload
    finally:
        proxy.stop()
        up.shutdown()
    assert sink._order == []


def test_proxy_dead_upstream_502():
    sink = TraceSink(model_family="m", source="t", license_="l")
    rec = OpenAIRecorder(sink, FakeTokenizer(), model_family="m")
    # Nothing listens on this port.
    proxy = ProxyServer("http://127.0.0.1:1", rec)
    proxy.start()
    try:
        status, _ = _post(
            f"http://127.0.0.1:{proxy.port}/v1/chat/completions", {"messages": MESSAGES}
        )
        assert status == 502
    finally:
        proxy.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_capture_requires_transformers():
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed")
    from state_bench.cli import main

    rc = main(
        [
            "capture",
            "--upstream",
            "http://127.0.0.1:1",
            "--tokenizer",
            "fake",
            "-o",
            "out.jsonl",
            "--license",
            "Apache-2.0",
        ]
    )
    assert rc == 2

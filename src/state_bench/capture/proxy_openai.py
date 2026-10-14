"""OpenAI-compatible recording proxy (capture source plugin).

Sits between an agent and any OpenAI-compatible server (first target:
SGLang). For each ``/v1/chat/completions`` request it:

* renders the prompt with the model tokenizer and chat template (the same
  rendering the server applies, generation prompt included),
* hashes the prompt token ids into chained 64-token blocks
  (:data:`base.HASH_SCHEME`),
* derives message ``segments`` from per-message template renderings,
  absorbing template glue (leading tokens and the generation-prompt tail)
  into the adjacent message; when the template cannot be attributed it
  records ``segments=None`` instead of guessing,
* records the measured times and the server-reported ``usage`` (sidecar
  annotation, including a local-vs-server prompt-length cross-check).

Lineage and pattern are inferred by the :class:`base.TraceSink`. A
conversation starts a new captured session when its request shares no full
block with anything seen before.

Standard library only for the transport (``http.server``, ``urllib``); the
tokenizer is loaded lazily from ``transformers`` (optional dependency group
``capture``).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..traces.schema import Segment
from .base import (
    CapturedRequest,
    TokenBlockHasher,
    TraceSink,
    shared_full_block_hashes,
)

CHAT_PATH = "/v1/chat/completions"
ROLE_TO_KIND = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool_result"}


class OpenAIRecorder:
    """Turns observed chat-completions traffic into :class:`CapturedRequest` values."""

    def __init__(
        self, sink: TraceSink, tokenizer: object, *, model_family: str, source: str = "proxy:openai"
    ) -> None:
        self.sink = sink
        self.tok = tokenizer
        self.model_family = model_family
        self.source = source
        self.lock = threading.Lock()
        self._t0: float | None = None
        self._seq = 0
        self._seen: list[tuple[str, tuple[str, ...]]] = []
        self._sess_start: dict[str, float] = {}

    # -- rendering ------------------------------------------------------------

    def _render(self, messages: list[dict], tools: object) -> list[int] | None:
        apply = self.tok.apply_chat_template
        try:
            if tools:
                return list(apply(messages, tools=tools, add_generation_prompt=True, tokenize=True))
            return list(apply(messages, add_generation_prompt=True, tokenize=True))
        except (TypeError, ValueError, KeyError) as e:
            print(f"state-bench capture: cannot render prompt: {e}", file=sys.stderr)
            return None

    def _segments(self, messages: list[dict], full_ids: list[int]) -> tuple[Segment, ...] | None:
        kinds = [ROLE_TO_KIND.get(m.get("role")) for m in messages]
        if not messages or any(k is None for k in kinds):
            return None
        spans: list[list[int]] = []
        prev = 0
        for i in range(len(messages)):
            try:
                ids = self.tok.apply_chat_template(
                    messages[: i + 1], add_generation_prompt=False, tokenize=True
                )
            except (TypeError, ValueError, KeyError):
                return None
            end = len(ids)
            if end <= prev:
                return None
            spans.append([prev, end])
            prev = end
        if prev > len(full_ids):
            return None
        # Absorb template glue: leading tokens into the first message, the
        # generation-prompt tail into the last one.
        spans[0][0] = 0
        spans[-1][1] = len(full_ids)
        return tuple(Segment(kinds[i], spans[i][0], spans[i][1]) for i in range(len(spans)))  # type: ignore[arg-type]

    # -- recording --------------------------------------------------------------

    def _session_for(self, blocks: tuple[str, ...], t: float) -> str:
        best: str | None = None
        best_shared = 0
        for sid, seen in self._seen:
            shared = shared_full_block_hashes(seen, blocks)
            if shared > best_shared or (shared == best_shared and shared > 0):
                best, best_shared = sid, shared
        if best is None:
            sid = f"c{self._seq:04d}"
            self._seq += 1
            self.sink.start_session(sid, round(t - self._t0, 6))  # type: ignore[operator]
            self._sess_start[sid] = t
            return sid
        return best

    def record(
        self,
        messages: object,
        tools: object,
        t_arrival: float,
        t_done: float,
        usage: object,
    ) -> None:
        """Record one completed request. Must not raise into the serving path."""
        if not isinstance(messages, list):
            return
        full_ids = self._render(messages, tools)
        if not full_ids:
            return
        hasher = TokenBlockHasher()
        hasher.extend(full_ids)
        blocks = hasher.blocks()
        segments = self._segments(messages, full_ids)
        annotations: dict[str, object] = {}
        if isinstance(usage, dict):
            annotations["usage"] = usage
            server = usage.get("prompt_tokens")
            if isinstance(server, int) and server != len(full_ids):
                annotations["prompt_tokens_local_vs_server"] = [len(full_ids), server]
        completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
        with self.lock:
            if self._t0 is None:
                self._t0 = t_arrival
            sid = self._session_for(blocks, t_arrival)
            self._seen.append((sid, blocks))
            base = self._sess_start[sid]
            self.sink.request(
                CapturedRequest(
                    session_id=sid,
                    t_arrival_s=round(t_arrival - base, 6),
                    t_done_s=round(t_done - base, 6),
                    prompt_tokens=len(full_ids),
                    blocks=blocks,
                    segments=segments,
                    output_tokens=completion if isinstance(completion, int) else 0,
                    annotations=annotations,
                )
            )


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _sse_usage(line: bytes, usage: object) -> object:
    line = line.strip()
    if not line.startswith(b"data:"):
        return usage
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return usage
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return usage
    u = obj.get("usage") if isinstance(obj, dict) else None
    return u if isinstance(u, dict) else usage


def _json_usage(payload: bytes) -> object:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    u = obj.get("usage") if isinstance(obj, dict) else None
    return u if isinstance(u, dict) else None


def _make_handler(upstream: str, recorder: OpenAIRecorder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # silence default logging
            pass

        def do_GET(self) -> None:
            self._passthrough("GET")

        def do_POST(self) -> None:
            if self.path == CHAT_PATH:
                self._record_chat()
            else:
                self._passthrough("POST")

        # -- helpers ------------------------------------------------------------

        def _read_body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _passthrough(self, method: str) -> None:
            body = self._read_body() if method == "POST" else None
            req = urllib.request.Request(
                upstream + self.path,
                data=body,
                method=method,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as up:
                    self.send_response(up.status)
                    self.send_header(
                        "Content-Type", up.headers.get("Content-Type", "application/json")
                    )
                    self.end_headers()
                    self.wfile.write(up.read())
            except urllib.error.HTTPError as e:
                self._send_bytes(e.code, e.read())
            except Exception as e:  # report, do not take the client down
                self.send_error(502, f"upstream error: {e}")

        def _send_bytes(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def _record_chat(self) -> None:
            body = self._read_body()
            t0 = time.monotonic()
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "invalid json")
                return
            streaming = bool(req.get("stream"))
            if streaming and isinstance(req, dict):
                # Ask the server to include usage in the final stream chunk so
                # output length is measured, not estimated.
                req["stream_options"] = {**(req.get("stream_options") or {}), "include_usage": True}
                body = json.dumps(req).encode()
            messages = req.get("messages") if isinstance(req, dict) else None
            tools = req.get("tools") if isinstance(req, dict) else None
            upstream_req = urllib.request.Request(
                upstream + CHAT_PATH,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            usage: object = None
            try:
                with urllib.request.urlopen(upstream_req) as up:
                    self.send_response(up.status)
                    self.send_header(
                        "Content-Type", up.headers.get("Content-Type", "application/json")
                    )
                    self.end_headers()
                    if streaming:
                        buf = b""
                        while True:
                            chunk = up.read1(65536)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            buf += chunk
                            while b"\n" in buf:
                                line, buf = buf.split(b"\n", 1)
                                usage = _sse_usage(line, usage)
                    else:
                        payload = up.read()
                        self.wfile.write(payload)
                        usage = _json_usage(payload)
                    t_done = time.monotonic()
            except urllib.error.HTTPError as e:
                self._send_bytes(e.code, e.read())
                return
            except Exception as e:
                self.send_error(502, f"upstream error: {e}")
                return
            try:
                recorder.record(messages, tools, t0, t_done, usage)
            except Exception as e:  # recording must never break serving
                print(f"state-bench capture: failed to record request: {e}", file=sys.stderr)

    return Handler


class ProxyServer:
    """A recording proxy in a daemon thread; ``port`` is known after ``start()``."""

    def __init__(
        self, upstream: str, recorder: OpenAIRecorder, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        handler = _make_handler(upstream.rstrip("/"), recorder)
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

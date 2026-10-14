"""Command-line entry point: ``state-bench <command>``.

Every command is deterministic and exits non-zero on failure. ``stats`` and
``schema`` print JSON so their output can be consumed by scripts and agents.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

from . import __version__
from .traces import schema, synth


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        trace = schema.load_trace(args.path, check=True)
    except schema.TraceError as e:
        print(f"invalid: {e}", file=sys.stderr)
        return 1
    print(
        f"valid: {len(trace.sessions)} sessions, {len(trace.requests)} requests "
        f"(schema {trace.header.schema_version})"
    )
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    trace = schema.load_trace(args.path, check=False)
    reqs = trace.requests
    prompt = [r.prompt_tokens for r in reqs]
    by_pattern: dict[str, int] = {}
    for r in reqs:
        key = r.pattern or "root"
        by_pattern[key] = by_pattern.get(key, 0) + 1
    idx = {r.req_id: r for r in reqs}
    reusable = sum(
        schema.shared_full_blocks(r, idx[r.parent_req_id]) * schema.BLOCK_TOKENS
        for r in reqs
        if r.parent_req_id
    )
    out = {
        "generator": trace.header.generator or "unknown",
        "sessions": len(trace.sessions),
        "requests": len(reqs),
        "prompt_tokens_total": sum(prompt),
        "prompt_tokens_median": statistics.median(prompt) if prompt else 0,
        "prompt_tokens_max": max(prompt, default=0),
        "patterns": by_pattern,
        # Upper bound for an ideal prefix cache that keeps every parent; not a measurement.
        "parent_reusable_token_ratio": round(reusable / sum(prompt), 4) if prompt else 0,
        "has_segments": all(r.segments is not None for r in reqs),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _slots(value: str) -> int | None:
    if value.lower() == "none":
        return None
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("state slots must be >= 0 or 'none'")
    return n


def _cmd_sim(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .sim import SimConfig, get_policy, simulate

    trace = schema.load_trace(args.path, check=False)
    base = dict(
        cache_chunk_size=args.cache_chunk_size,
        state_chunk_size=args.state_chunk_size,
        page_size=args.page_size,
        chunked_prefill_size=args.chunked_prefill_size,
        track_interval=args.track_interval,
        output_roundtrip=not args.no_output_roundtrip,
        first_chunk_cut=args.first_chunk_cut,
    )
    try:
        policies = [get_policy(name) for name in (args.policy or ["sglang_current"])]
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    rows = []
    for pol in policies:
        for n in args.state_slots or [None]:
            res = simulate(trace, pol, SimConfig(**base, state_slots=n))
            row = res.summary()
            if args.per_request:
                row["records"] = [asdict(r) for r in res.records]
            rows.append(row)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _cmd_schema(_: argparse.Namespace) -> int:
    print(json.dumps(schema.JSON_SCHEMA, indent=2))
    return 0


def _probe_model_family(upstream: str) -> str:
    try:
        with urllib.request.urlopen(upstream.rstrip("/") + "/v1/models", timeout=5) as r:
            obj = json.loads(r.read())
        models = obj.get("data")
        if models and isinstance(models[0].get("id"), str):
            return models[0]["id"]
    except Exception:  # best effort
        pass
    return "unknown"


def _cmd_capture(args: argparse.Namespace) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "capture requires the transformers dependency: uv pip install 'state-bench[capture]'",
            file=sys.stderr,
        )
        return 2
    from .capture import OpenAIRecorder, ProxyServer, TraceSink

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_family = args.model_family or _probe_model_family(args.upstream)
    sink = TraceSink(model_family=model_family, source="proxy:openai", license_=args.license)
    recorder = OpenAIRecorder(sink, tokenizer, model_family=model_family)
    server = ProxyServer(args.upstream, recorder, host=args.host, port=args.port)
    server.start()
    print(
        f"capturing {args.upstream} via http://{args.host}:{server.port} -> {args.output} "
        f"(Ctrl-C to stop)",
        file=sys.stderr,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    sink.finish(args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="state-bench",
        description="Traces and tools for evaluating state caching in LLM serving.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a trace's structure and semantics")
    v.add_argument("path")
    v.set_defaults(func=_cmd_validate)

    s = sub.add_parser("stats", help="summarize a trace as JSON")
    s.add_argument("path")
    s.set_defaults(func=_cmd_stats)

    sm = sub.add_parser(
        "sim", help="simulate state-snapshot policies on a trace (JSON, provenance: simulated)"
    )
    sm.add_argument("path")
    sm.add_argument(
        "--policy",
        action="append",
        help="policy name, optionally with parameters (pin_recent:k=2); repeat to compare",
    )
    sm.add_argument(
        "--state-slots",
        action="append",
        type=_slots,
        help="state pool size in snapshots, or 'none' for unlimited; repeat to sweep",
    )
    sm.add_argument("--cache-chunk-size", type=int, default=64)
    sm.add_argument("--state-chunk-size", type=int, default=64)
    sm.add_argument("--page-size", type=int, default=1)
    sm.add_argument("--chunked-prefill-size", type=int, default=2048)
    sm.add_argument("--track-interval", type=int, default=256)
    sm.add_argument(
        "--first-chunk-cut",
        choices=["none", "hashed"],
        default="none",
        help="'hashed' cuts the first of several prefill chunks at a per-request length",
    )
    sm.add_argument(
        "--no-output-roundtrip",
        action="store_true",
        help="assume replies are not replayed token for token in the next request",
    )
    sm.add_argument("--per-request", action="store_true", help="include per-request records")
    sm.set_defaults(func=_cmd_sim)

    sc = sub.add_parser("schema", help="print the JSON Schema for trace records")
    sc.set_defaults(func=_cmd_schema)

    sub.add_parser(
        "synth", help="generate a synthetic trace (see `state-bench synth -h`)", add_help=False
    )

    c = sub.add_parser(
        "capture",
        help="record a trace by proxying an OpenAI-compatible server (provenance: measured)",
    )
    c.add_argument("--upstream", required=True, help="base URL of the serving system")
    c.add_argument("--tokenizer", required=True, help="HF tokenizer path or model id")
    c.add_argument("-o", "--output", required=True, help="trace output path (JSONL)")
    c.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    c.add_argument("--port", type=int, default=0, help="bind port (default: ephemeral)")
    c.add_argument("--license", required=True, help="license statement for the captured data")
    c.add_argument(
        "--model-family", default=None, help="override model_family (default: probe /v1/models)"
    )
    c.set_defaults(func=_cmd_capture)

    args, rest = p.parse_known_args(argv)
    if args.cmd == "synth":
        return synth.main(rest)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

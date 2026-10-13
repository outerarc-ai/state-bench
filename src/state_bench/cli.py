"""Command-line entry point: ``state-bench <command>``.

Every command is deterministic and exits non-zero on failure. ``stats`` and
``schema`` print JSON so their output can be consumed by scripts and agents.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

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


def _cmd_schema(_: argparse.Namespace) -> int:
    print(json.dumps(schema.JSON_SCHEMA, indent=2))
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

    sc = sub.add_parser("schema", help="print the JSON Schema for trace records")
    sc.set_defaults(func=_cmd_schema)

    sub.add_parser(
        "synth", help="generate a synthetic trace (see `state-bench synth -h`)", add_help=False
    )

    args, rest = p.parse_known_args(argv)
    if args.cmd == "synth":
        return synth.main(rest)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

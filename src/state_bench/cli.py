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
        max_states_per_path=args.max_states_per_path,
        path_cap_min_usage=args.path_cap_min_usage,
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


def _cmd_import(args: argparse.Namespace) -> int:
    from .traces import weka

    try:
        n_sessions, n_requests = weka.write_import(
            args.paths,
            args.output,
            source=args.source,
            license_=args.license,
            session_interval_s=args.session_interval,
            limit=args.limit,
            generator=f"state-bench {__version__} import weka",
        )
    except (schema.TraceError, KeyError, ValueError) as e:
        print(f"import failed: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}: {n_sessions} sessions, {n_requests} requests", file=sys.stderr)
    return 0


def _cmd_paired(args: argparse.Namespace) -> int:
    from . import bench

    if args.action == "plan":
        for r in bench.plan(args.pairs, args.order):
            print(f"{r.run}\tpair {r.pair}\t{r.arm}")
        return 0
    if args.action == "load":
        from .bench.sglang_load import LoadShape, run_load

        shape = LoadShape(
            sessions=args.sessions,
            turns=args.turns,
            shared_prefix=args.shared_prefix,
            turn_min=args.turn_min,
            turn_max=args.turn_max,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        failures = run_load(args.url, args.output, shape)
        print(f"wrote {args.output}; {failures} failed requests", file=sys.stderr)
        return 0
    if args.action == "run":
        config = json.loads(open(args.config, encoding="utf-8").read())
        bench.run(config, args.output, log=lambda m: print(m, file=sys.stderr))
        return 0
    rep = bench.report(args.run_dir)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["complete_pairs"] > 0 else 1


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
        "--max-states-per-path",
        type=int,
        default=None,
        help="per-path snapshot cap, as SGLang's --mamba-max-states-per-path (default: none)",
    )
    sm.add_argument(
        "--path-cap-min-usage",
        type=float,
        default=0.0,
        help="apply the per-path cap only while the state pool is at least this full (0-1)",
    )
    sm.add_argument(
        "--no-output-roundtrip",
        action="store_true",
        help="assume replies are not replayed token for token in the next request",
    )
    sm.add_argument("--per-request", action="store_true", help="include per-request records")
    sm.set_defaults(func=_cmd_sim)

    im = sub.add_parser(
        "import", help="convert public traces to the state-bench format (see docs/PUBLIC_TRACES.md)"
    )
    im.add_argument("format", choices=["weka"], help="source format (kv-cache-tester, AgentX)")
    im.add_argument("paths", nargs="+", help="JSON or JSONL files, or directories of them")
    im.add_argument("-o", "--output", required=True, help="output JSONL trace")
    im.add_argument("--source", required=True, help="corpus name recorded in every session")
    im.add_argument("--license", required=True, help="corpus license recorded in every session")
    im.add_argument(
        "--session-interval",
        type=float,
        default=0.0,
        help="start session i at i * this many seconds (an assumption; default 0)",
    )
    im.add_argument("--limit", type=int, help="import at most this many traces")
    im.set_defaults(func=_cmd_import)

    pa = sub.add_parser(
        "paired", help="alternating baseline/candidate runs and their paired report"
    )
    pas = pa.add_subparsers(dest="action", required=True)
    pp = pas.add_parser("plan", help="print the run order")
    pp.add_argument("--pairs", type=int, default=5)
    pp.add_argument("--order", choices=["ab", "abba"], default="ab")
    pr = pas.add_parser("run", help="execute a run configuration (JSON)")
    pr.add_argument("config")
    pr.add_argument("-o", "--output", required=True, help="run directory to create")
    pl = pas.add_parser(
        "load", help="send a deterministic multi-turn load to an SGLang server; write records"
    )
    pl.add_argument("url", help="server base URL, e.g. http://127.0.0.1:30000")
    pl.add_argument("-o", "--output", required=True, help="records JSONL")
    pl.add_argument("--sessions", type=int, default=8)
    pl.add_argument("--turns", type=int, default=6)
    pl.add_argument("--shared-prefix", type=int, default=1024)
    pl.add_argument("--turn-min", type=int, default=256)
    pl.add_argument("--turn-max", type=int, default=1024)
    pl.add_argument("--max-new-tokens", type=int, default=16)
    pl.add_argument("--seed", type=int, default=17)
    prp = pas.add_parser("report", help="paired report of a run directory (JSON)")
    prp.add_argument("run_dir")
    pa.set_defaults(func=_cmd_paired)

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

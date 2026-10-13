"""Builders for hand-checked simulator traces."""

from state_bench.sim import Policy, SimConfig, simulate
from state_bench.traces.schema import Request, Session, Trace, TraceHeader
from state_bench.traces.synth import BYTES_PER_TOKEN, PrefixHasher


def toks(tag: int, n: int) -> list[int]:
    """``n`` tokens unique to ``tag``."""
    return [tag * 1_000_000 + i for i in range(n)]


def blocks_of(tokens: list[int]) -> tuple[str, ...]:
    h = PrefixHasher()
    h.extend(b"".join(t.to_bytes(BYTES_PER_TOKEN, "little") for t in tokens))
    return h.blocks()


def make_trace(*specs: tuple, times=None, sessions=None, segments=None) -> Trace:
    """Specs are (req_id, tokens, output_tokens, parent_req_id, pattern), in arrival order.

    Request ``i`` arrives at ``times[i]`` seconds (default ``i``) in session
    ``sessions[i]`` (default ``"s"``), with message boundaries ``segments[i]``.
    """
    reqs = [
        Request(
            req_id=rid,
            session_id=sessions[i] if sessions else "s",
            parent_req_id=parent,
            branch_id=0,
            t_arrival_s=float(times[i] if times else i),
            prompt_tokens=len(tokens),
            blocks=blocks_of(tokens),
            segments=segments[i] if segments else None,
            output_tokens=out,
            tool_wait_s=None,
            pattern=pattern,
        )
        for i, (rid, tokens, out, parent, pattern) in enumerate(specs)
    ]
    ids = sorted({r.session_id for r in reqs})
    return Trace(TraceHeader(), [Session(x, 0.0, "test", "test", "Apache-2.0") for x in ids], reqs)


def req(tokens: list[int], out: int = 1) -> Request:
    return make_trace(("X", tokens, out, None, None)).requests[0]


def run(trace: Trace, policy: Policy, **cfg):
    res = simulate(trace, policy, SimConfig(**cfg))
    return res, {r.req_id: r for r in res.records}

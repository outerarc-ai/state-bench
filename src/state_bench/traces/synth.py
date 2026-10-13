"""Synthetic trace generator.

Every parameter here is an **assumption**. Synthetic traces are for unit-level
policy tests and sensitivity analysis; they are not a basis for performance claims.

Four request patterns, each relative to a parent request:

* ``append``: parent input + parent output + a new tool result. This is the main
  loop of an agent.
* ``fork``: a sub-agent branches off the main chain. The branch shares either the
  system prompt (``share="system"``) or the parent's whole input (``share="full"``),
  adds a task message, then appends on its own. The parent resumes after every
  branch finishes, as if the sub-agent were one tool call. ``width`` sets how many
  branches run concurrently.
* ``retry``: the same input is sent again (timeout or error). Later appends continue
  from the retry.
* ``rewrite``: history is rewritten. The first ``keep_messages`` messages are kept
  and the rest is replaced by a summary (context compaction or an edited earlier
  message). It can also trigger automatically when the context limit is exceeded.

Content model: a message is only a seed and a length. Its bytes are expanded
deterministically with ``shake_256(seed)`` and hashed in 64-token chained blocks,
so a message that reappears in a later request produces the same block hashes.
Prefix sharing therefore follows from construction.

Timing model (also an assumption): service time is
``nominal_ttft_s + output_tokens / nominal_decode_tps``; the next request arrives
after the service time plus ``tool_wait_s``. When replaying against a real server,
use ``tool_wait_s`` (closed loop); ``t_arrival_s`` is nominal.

Current simplifications: branches neither fork nor rewrite; when a request is
retried, ``tool_wait_s`` is recorded on the retry and the original's is null.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import random
import struct
from dataclasses import asdict, dataclass, field
from typing import Literal

from .schema import (
    BLOCK_TOKENS,
    Request,
    Segment,
    Session,
    Trace,
    TraceHeader,
    dump_trace,
    validate_trace,
)

BYTES_PER_TOKEN = 4
GENERATOR_VERSION = "synth-0.2"
HASH_SCHEME = "blake2b64-chain-v1"
PROVENANCE = "assumption"


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LenDist:
    """Distribution of a token length or count. ``fixed`` is meant for hand-checked tests."""

    kind: Literal["fixed", "lognormal"] = "fixed"
    median: int = 256
    sigma: float = 0.0
    lo: int = 1
    hi: int = 1_000_000

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            v = self.median
        else:
            v = round(rng.lognormvariate(math.log(self.median), self.sigma))
        return max(self.lo, min(self.hi, v))


@dataclass(frozen=True)
class TimeDist:
    """Distribution of a duration in seconds: fixed, or lognormal by median and sigma."""

    kind: Literal["fixed", "lognormal"] = "fixed"
    median_s: float = 1.0
    sigma: float = 0.0

    def sample(self, rng: random.Random) -> float:
        if self.kind == "fixed":
            return self.median_s
        return rng.lognormvariate(math.log(self.median_s), self.sigma)


# ---------------------------------------------------------------------------
# Parameters (all assumptions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppendParams:
    """Main-chain parameters. Defaults sketch a coding agent; all are assumptions."""

    turns: LenDist = LenDist("lognormal", median=20, sigma=0.6, lo=2, hi=200)
    """Requests on the main chain, including the first and excluding retries."""
    system_tokens: LenDist = LenDist("fixed", median=6000)
    """System prompt plus tool definitions."""
    shared_system_prompt: bool = True
    """All sessions use the same system prompt, so sessions can share a prefix."""
    user_tokens: LenDist = LenDist("lognormal", median=300, sigma=0.8, lo=16, hi=8000)
    """First user message."""
    output_tokens: LenDist = LenDist("lognormal", median=250, sigma=0.9, lo=8, hi=8000)
    """Each assistant output (a tool call or a reply)."""
    tool_result_tokens: LenDist = LenDist("lognormal", median=1200, sigma=1.1, lo=8, hi=30000)
    """Each tool result."""
    tool_wait: TimeDist = TimeDist("lognormal", median_s=2.0, sigma=1.0)
    """Tool execution time, from the end of an output to the next arrival."""
    max_prompt_tokens: int = 131072
    """Context limit. Beyond it, a chain ends unless ``rewrite.on_overflow`` compacts it."""
    nominal_ttft_s: float = 0.5
    nominal_decode_tps: float = 50.0


@dataclass(frozen=True)
class ForkParams:
    prob: float = 0.0
    """Probability of spawning sub-agents after each main-chain request."""
    width: LenDist = LenDist("fixed", median=1)
    """Number of concurrent branches per spawn."""
    share: Literal["system", "full"] = "system"
    """Whether a branch shares only the system prompt or the parent's whole input."""
    task_tokens: LenDist = LenDist("lognormal", median=400, sigma=0.7, lo=16, hi=8000)
    """Task message that starts each branch."""
    turns: LenDist = LenDist("lognormal", median=8, sigma=0.6, lo=1, hi=100)
    """Requests per branch."""
    start_delay_s: float = 0.05
    """Gap between the end of the parent's output and the first branch request."""


@dataclass(frozen=True)
class RetryParams:
    prob: float = 0.0
    """Probability that a request is sent again unchanged (at most once per request)."""
    delay: TimeDist = TimeDist("lognormal", median_s=5.0, sigma=0.8)
    """Gap between the original arrival and the retry arrival."""


@dataclass(frozen=True)
class RewriteParams:
    prob: float = 0.0
    """Probability of rewriting history after each main-chain request."""
    keep_messages: int = 2
    """Messages kept at the front (2 = system prompt + first user message)."""
    summary_tokens: LenDist = LenDist("lognormal", median=2000, sigma=0.5, lo=64, hi=16000)
    """Summary that replaces the rest."""
    on_overflow: bool = False
    """Rewrite (compact) instead of ending the session when the context limit is exceeded."""


PRESETS = ("append", "fork", "retry", "rewrite", "mixed")


@dataclass(frozen=True)
class TraceParams:
    seed: int = 0
    sessions: int = 8
    session_interarrival: TimeDist = TimeDist("lognormal", median_s=30.0, sigma=1.0)
    model_family: str = "qwen3.5"
    preset: str = "append"
    """Label recorded in ``session.source``; behavior comes from the parameter groups."""
    append: AppendParams = field(default_factory=AppendParams)
    fork: ForkParams = field(default_factory=ForkParams)
    retry: RetryParams = field(default_factory=RetryParams)
    rewrite: RewriteParams = field(default_factory=RewriteParams)


def preset_params(preset: str, **kw) -> TraceParams:
    """Parameter presets used by the CLI. Pattern probabilities are assumptions."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
    f = ForkParams(prob=0.15) if preset in ("fork", "mixed") else ForkParams()
    r = RetryParams(prob=0.05) if preset in ("retry", "mixed") else RetryParams()
    w = (
        RewriteParams(prob=0.03, on_overflow=True)
        if preset in ("rewrite", "mixed")
        else RewriteParams()
    )
    return TraceParams(preset=preset, fork=f, retry=r, rewrite=w, **kw)


# ---------------------------------------------------------------------------
# Content and hashing
# ---------------------------------------------------------------------------


def _derive_seed(*parts: object) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return h.digest()


def message_bytes(seed: bytes, n_tokens: int) -> bytes:
    """Deterministic stand-in content for a message: ``n_tokens * 4`` bytes."""
    return hashlib.shake_256(seed).digest(n_tokens * BYTES_PER_TOKEN)


def _block_hash(prev: bytes, n_tokens: int, payload: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=8)
    h.update(prev)
    h.update(struct.pack("<H", n_tokens))  # distinguishes a partial tail from a full block
    h.update(payload)
    return h.digest()


_ZERO = b"\x00" * 8


class PrefixHasher:
    """Chained 64-token block hashing over a token stream, caching full blocks."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._full: list[bytes] = []

    @property
    def n_tokens(self) -> int:
        return len(self._buf) // BYTES_PER_TOKEN

    def extend(self, payload: bytes) -> None:
        self._buf += payload
        bb = BLOCK_TOKENS * BYTES_PER_TOKEN
        while (len(self._full) + 1) * bb <= len(self._buf):
            i = len(self._full)
            prev = self._full[-1] if self._full else _ZERO
            self._full.append(
                _block_hash(prev, BLOCK_TOKENS, bytes(self._buf[i * bb : (i + 1) * bb]))
            )

    def truncated(self, n_tokens: int) -> PrefixHasher:
        """Copy of the first ``n_tokens`` tokens, reusing cached full-block hashes."""
        if not 0 <= n_tokens <= self.n_tokens:
            raise ValueError(n_tokens)
        out = PrefixHasher()
        out._buf = bytearray(self._buf[: n_tokens * BYTES_PER_TOKEN])
        out._full = self._full[: n_tokens // BLOCK_TOKENS]
        return out

    def blocks(self) -> tuple[str, ...]:
        """Block hashes of the current input, including a partial tail."""
        out = [b.hex() for b in self._full]
        tail_tokens = self.n_tokens - len(self._full) * BLOCK_TOKENS
        if tail_tokens:
            prev = self._full[-1] if self._full else _ZERO
            tail = bytes(self._buf[len(self._full) * BLOCK_TOKENS * BYTES_PER_TOKEN :])
            out.append(_block_hash(prev, tail_tokens, tail).hex())
        return tuple(out)


class _Context:
    """The current input of one chain: block hashes plus message boundaries."""

    def __init__(
        self, hasher: PrefixHasher | None = None, segs: list[Segment] | None = None
    ) -> None:
        self.hasher = hasher or PrefixHasher()
        self.segs: list[Segment] = segs or []

    @property
    def n_tokens(self) -> int:
        return self.hasher.n_tokens

    def add(self, kind: str, seed: bytes, n: int) -> None:
        start = self.hasher.n_tokens
        self.hasher.extend(message_bytes(seed, n))
        self.segs.append(Segment(kind, start, start + n))  # type: ignore[arg-type]

    def keep(self, n_messages: int) -> _Context:
        """Copy holding only the first ``n_messages`` messages."""
        n_messages = max(1, min(n_messages, len(self.segs)))
        end = self.segs[n_messages - 1].end_tok
        return _Context(self.hasher.truncated(end), list(self.segs[:n_messages]))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class _SessionGen:
    def __init__(self, sid: str, params: TraceParams, rng: random.Random) -> None:
        self.sid = sid
        self.p = params
        self.rng = rng
        self.reqs: list[tuple[float, int, Request]] = []  # (arrival, creation order, request)
        self._seq = 0
        self._next_branch = 1

    def seed(self, *parts: object) -> bytes:
        return _derive_seed(self.p.seed, self.sid, *parts)

    def emit(
        self,
        ctx: _Context,
        t: float,
        parent: Request | None,
        pattern: str | None,
        branch: int,
        out_tok: int,
    ) -> int:
        r = Request(
            req_id=f"{self.sid}-r{self._seq:04d}",
            session_id=self.sid,
            parent_req_id=parent.req_id if parent else None,
            branch_id=branch,
            t_arrival_s=round(t, 6),
            prompt_tokens=ctx.n_tokens,
            blocks=ctx.hasher.blocks(),
            segments=tuple(ctx.segs),
            output_tokens=out_tok,
            tool_wait_s=None,
            pattern=None if parent is None else pattern,  # type: ignore[arg-type]
        )
        self.reqs.append((t, self._seq, r))
        self._seq += 1
        return len(self.reqs) - 1

    def set_wait(self, idx: int, wait: float) -> None:
        t, s, r = self.reqs[idx]
        self.reqs[idx] = (t, s, dataclasses.replace(r, tool_wait_s=round(wait, 6)))

    def service_s(self, out_tok: int) -> float:
        ap = self.p.append
        return ap.nominal_ttft_s + out_tok / ap.nominal_decode_tps

    def chain(
        self,
        ctx: _Context,
        t: float,
        parent: Request | None,
        first_pattern: str | None,
        branch: int,
        n_turns: int,
        *,
        main: bool,
    ) -> float:
        """Generate one chain and return when its last output ends."""
        ap, rng = self.p.append, self.rng
        pattern = first_pattern
        end_t = t
        for i in range(n_turns):
            out_tok = ap.output_tokens.sample(rng)
            idx = self.emit(ctx, t, parent, pattern, branch, out_tok)
            cur = self.reqs[idx][2]

            if rng.random() < self.p.retry.prob:
                t = t + self.p.retry.delay.sample(rng)
                out_tok = ap.output_tokens.sample(rng)
                idx = self.emit(ctx, t, cur, "retry", branch, out_tok)
                cur = self.reqs[idx][2]

            out_end = t + self.service_s(out_tok)
            end_t = out_end
            if i == n_turns - 1:
                break

            wait = ap.tool_wait.sample(rng)
            if main and rng.random() < self.p.fork.prob:
                branch_end = self.spawn(ctx, cur, out_end)
                wait += branch_end - out_end  # the parent resumes after its sub-agents

            tool_tok = ap.tool_result_tokens.sample(rng)
            grow = ctx.n_tokens + out_tok + tool_tok
            rw = self.p.rewrite
            do_rewrite = main and (
                rng.random() < rw.prob or (rw.on_overflow and grow > ap.max_prompt_tokens)
            )
            if do_rewrite:
                ctx = ctx.keep(rw.keep_messages)
                ctx.add("user", self.seed("summary", branch, i), rw.summary_tokens.sample(rng))
                pattern = "rewrite"
                if ctx.n_tokens > ap.max_prompt_tokens:
                    break
            elif grow > ap.max_prompt_tokens:
                break
            else:
                ctx.add("assistant", self.seed("asst", branch, i), out_tok)
                ctx.add("tool_result", self.seed("tool", branch, i), tool_tok)
                pattern = "append"

            self.set_wait(idx, wait)
            t = out_end + wait
            parent = cur
        return end_t

    def spawn(self, ctx: _Context, parent: Request, out_end: float) -> float:
        """Start concurrent branches after ``parent`` and return the latest end time."""
        fp, rng = self.p.fork, self.rng
        latest = out_end
        for _ in range(fp.width.sample(rng)):
            b = self._next_branch
            self._next_branch += 1
            base = ctx.keep(1) if fp.share == "system" else ctx.keep(len(ctx.segs))
            base.add("user", self.seed("task", b), fp.task_tokens.sample(rng))
            end = self.chain(
                base,
                out_end + fp.start_delay_s,
                parent,
                "fork",
                b,
                fp.turns.sample(rng),
                main=False,
            )
            latest = max(latest, end)
        return latest

    def run(self) -> list[Request]:
        ap = self.p.append
        ctx = _Context()
        if ap.shared_system_prompt:
            sys_seed = _derive_seed(self.p.seed, "system")
            sys_len = ap.system_tokens.sample(
                random.Random(_derive_seed(self.p.seed, "system-len"))
            )
        else:
            sys_seed = self.seed("system")
            sys_len = ap.system_tokens.sample(self.rng)
        ctx.add("system", sys_seed, sys_len)
        ctx.add("user", self.seed("user"), ap.user_tokens.sample(self.rng))
        self.chain(ctx, 0.0, None, None, 0, ap.turns.sample(self.rng), main=True)
        self.reqs.sort(key=lambda x: (x[0], x[1]))
        return [r for _, _, r in self.reqs]


def generate_trace(params: TraceParams) -> Trace:
    """Generate a synthetic trace. Identical parameters, seed included, give identical bytes."""
    rng = random.Random(params.seed)
    sessions: list[Session] = []
    requests: list[Request] = []
    t0 = 0.0
    for k in range(params.sessions):
        sid = f"s{k:04d}"
        if k > 0:
            t0 += params.session_interarrival.sample(rng)
        sessions.append(
            Session(
                session_id=sid,
                t_start_s=round(t0, 6),
                model_family=params.model_family,
                source=f"synth:{params.preset}",
                license="Apache-2.0",
            )
        )
        srng = random.Random(_derive_seed(params.seed, sid))
        requests.extend(_SessionGen(sid, params, srng).run())

    header = TraceHeader(
        hash_scheme=HASH_SCHEME,
        generator=GENERATOR_VERSION,
        params={"provenance": PROVENANCE, **asdict(params)},
    )
    return Trace(header=header, sessions=sessions, requests=requests)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="state-bench synth",
        description="Generate a synthetic trace. All parameters are assumptions.",
    )
    ap.add_argument("--preset", default="append", choices=PRESETS)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-shared-system", action="store_true", help="give each session its own system prompt"
    )
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)

    params = preset_params(
        args.preset,
        seed=args.seed,
        sessions=args.sessions,
        append=AppendParams(shared_system_prompt=not args.no_shared_system),
    )
    trace = generate_trace(params)
    errs = validate_trace(trace)
    if errs:
        raise SystemExit(
            "generated trace failed validation (generator bug):\n" + "\n".join(errs[:10])
        )
    dump_trace(trace, args.output)
    n_tok = sum(r.prompt_tokens for r in trace.requests)
    print(
        f"wrote {args.output}: {len(trace.sessions)} sessions, "
        f"{len(trace.requests)} requests, {n_tok} input tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

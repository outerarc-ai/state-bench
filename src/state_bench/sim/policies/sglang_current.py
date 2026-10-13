"""SGLang's current snapshot placement (``extra_buffer`` strategy).

Implements rules R2 and R5 to R13 of ``docs/SGLANG_MAMBA_CHECKPOINTS.md``, read from SGLang
commit ``166b4ae7f``. Eviction (R14) is the simulator's state-pool LRU; the per-path cap
(R15) is left at its default of unlimited.
"""

from __future__ import annotations

from ...traces.schema import Request
from ..core import SimConfig


class SGLangCurrent:
    """Snapshot positions as SGLang chooses them today."""

    name = "sglang_current"

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        grid = cfg.checkpoint_grid
        # R2: last grid point inside the full-KV hit, if beyond the usable prefix.
        branch = (kv_hit // grid) * grid
        branching = branch if branch > start else None

        out: list[int] = []
        pos = start
        while pos < req.prompt_tokens:
            # R8: chunks are measured from the matched prefix.
            end = min(pos + cfg.chunked_prefill_size, req.prompt_tokens)
            ext = end - pos
            # R5/R6: one snapshot per chunk, aligned relative to the chunk start.
            if ext >= grid:
                track = pos + (ext // grid) * grid
                # The engine passes the chunk end as the track limit, or track + 1 when the
                # grid point is not a model chunk boundary (schedule_batch.py:3063-3071, 3093).
                model_aligned = pos + (ext // cfg.state_chunk_size) * cfg.state_chunk_size
                limit = end if model_aligned == track else track + 1
                # R7: the branching point replaces the default when it is inside the chunk
                # and aligned to cache_chunk_size relative to the chunk start.
                if (
                    branching is not None
                    and pos < branching < limit
                    and (branching - pos) % cfg.cache_chunk_size == 0
                ):
                    track = branching
                # R9/R10: each chunk inserts its snapshot.
                out.append(track)
            pos = end
        return out

    def plan_finish(self, req: Request, cfg: SimConfig) -> int | None:
        # R12/R13: decode records a state at each track_grid multiple the sequence reaches.
        # Decode processes tokens up to prompt + output - 1; the final token is never fed back.
        last = req.prompt_tokens + req.output_tokens - 1
        m = (last // cfg.track_grid) * cfg.track_grid
        return m if m > req.prompt_tokens else None

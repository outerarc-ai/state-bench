"""SGLang's current snapshot placement (``extra_buffer`` strategy).

Implements rules R1a, R2, and R5 to R13 of ``docs/SGLANG_MAMBA_CHECKPOINTS.md``, read
from SGLang commit ``166b4ae7f``. Eviction (R14) is least-recently-used; the per-path cap
(R15) is left at its default of unlimited.

``absolute_mask=True`` applies the donation mask SGLang uses when DCP is enabled: a chunk stores ``floor(chunk_end / grid) * grid`` only when
that position is a whole number of ``cache_chunk_size`` steps from the chunk start
(``schedule_batch.py:3035-3042``). Between SGLang #39115 and #40907 this mask also
applied without DCP. Other DCP effects, such as wider radix pages, are not modeled.
"""

from __future__ import annotations

from ...traces.schema import Request
from ..core import Policy, SimConfig, chunk_bounds


class SGLangCurrent(Policy):
    """Snapshot positions as SGLang chooses them."""

    name = "sglang_current"

    def __init__(self, absolute_mask: bool = False) -> None:
        self.absolute_mask = absolute_mask
        if absolute_mask:
            self.name = "sglang_absolute_mask"

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        grid = cfg.checkpoint_grid
        # R2: last grid point inside the full-KV hit, if beyond the usable prefix.
        branch = (kv_hit // grid) * grid
        branching = branch if branch > start else None

        out: list[int] = []
        # R8: chunks are measured from the matched prefix.
        for pos, end in chunk_bounds(req, start, cfg):
            ext = end - pos
            if self.absolute_mask:
                track = (end // grid) * grid
                mask = track > pos and (track - pos) % cfg.cache_chunk_size == 0
            else:
                # R5/R6: one snapshot per chunk, aligned relative to the chunk start.
                track = pos + (ext // grid) * grid
                mask = ext >= grid
            if not mask:
                continue
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
        return out

    def plan_finish(self, req: Request, cfg: SimConfig) -> list[int]:
        # R12/R13: decode records a state at each track_grid multiple the sequence reaches.
        # Decode processes tokens up to prompt + output - 1; the final token is never fed back.
        last = req.prompt_tokens + req.output_tokens - 1
        m = (last // cfg.track_grid) * cfg.track_grid
        return [m] if m > req.prompt_tokens else []

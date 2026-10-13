"""Placement policies taken from other engines' merged changes.

Each policy reproduces the placement rule as its source describes it, on the simulator's
checkpoint grid. Eviction is least-recently-used for both.
"""

from __future__ import annotations

from ...traces.schema import Request
from ..core import Policy, SimConfig


class PromptEndAnchor(Policy):
    """One snapshot at the end of each prompt, rounded down to the checkpoint grid.

    Follows the prompt-end anchor of ATOM #2045 (merged) with its interval and demand
    checkpoints turned off, the configuration its author used for the reported Kimi-K3
    measurements. No snapshots are taken during decode.
    """

    name = "prompt_end_anchor"

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        anchor = (req.prompt_tokens // cfg.checkpoint_grid) * cfg.checkpoint_grid
        return [anchor] if anchor > start else []


class MarconiAdmission(Policy):
    """Last state plus shared-prefix admission, as in vLLM #37898 (merged).

    The request's last state is stored at the last grid point of everything it computed,
    prompt and reply. When the KV hit reaches past the usable state, which reveals a
    shared prefix, a snapshot is also stored at the last grid point within the KV hit.
    This follows the vLLM implementation of the admission rule from Marconi (Pan et al.,
    MLSys 2025), not Marconi's FLOP-aware eviction.
    """

    name = "marconi_admission"

    def plan_prefill(self, req: Request, start: int, kv_hit: int, cfg: SimConfig) -> list[int]:
        shared = (kv_hit // cfg.checkpoint_grid) * cfg.checkpoint_grid
        return [shared] if shared > start else []

    def plan_finish(self, req: Request, cfg: SimConfig) -> list[int]:
        last = req.prompt_tokens + req.output_tokens - 1
        return [(last // cfg.checkpoint_grid) * cfg.checkpoint_grid]

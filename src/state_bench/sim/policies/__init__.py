"""Snapshot placement and retention policies, looked up by name.

A policy spec is a name with optional parameters, for example ``pin_recent`` or
``pin_recent:k=2,active_s=none``. Values are parsed as integers, floats, ``true``,
``false``, or ``none``.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core import Policy
from .boundary_value import BoundaryValue
from .oracle import OfflineOracle
from .pin_recent import PinRecent
from .sglang_current import SGLangCurrent
from .simple import MarconiAdmission, PromptEndAnchor

POLICIES: dict[str, Callable[..., Policy]] = {
    "sglang_current": SGLangCurrent,
    "sglang_absolute_mask": lambda: SGLangCurrent(absolute_mask=True),
    "prompt_end_anchor": PromptEndAnchor,
    "marconi_admission": MarconiAdmission,
    "pin_recent": PinRecent,
    "boundary_value": BoundaryValue,
    "offline_oracle": OfflineOracle,
}


def _parse_value(text: str) -> object:
    low = text.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def get_policy(spec: str) -> Policy:
    """Build a policy from ``name`` or ``name:key=value,...``."""
    name, _, params = spec.partition(":")
    if name not in POLICIES:
        known = ", ".join(sorted(POLICIES))
        raise ValueError(f"unknown policy {name!r}; known: {known}")
    kwargs = {}
    for item in filter(None, params.split(",")):
        key, eq, value = item.partition("=")
        if not eq:
            raise ValueError(f"policy parameter {item!r} must be key=value")
        kwargs[key.strip()] = _parse_value(value.strip())
    try:
        policy = POLICIES[name](**kwargs)
    except TypeError as e:
        raise ValueError(f"bad parameters for policy {name!r}: {e}") from None
    if kwargs:
        policy.name = spec
    return policy


__all__ = [
    "POLICIES",
    "BoundaryValue",
    "MarconiAdmission",
    "OfflineOracle",
    "PinRecent",
    "PromptEndAnchor",
    "SGLangCurrent",
    "get_policy",
]

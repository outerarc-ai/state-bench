"""Snapshot placement policies, looked up by name."""

from __future__ import annotations

from ..core import Policy
from .sglang_current import SGLangCurrent

POLICIES: dict[str, type] = {
    SGLangCurrent.name: SGLangCurrent,
}


def get_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        known = ", ".join(sorted(POLICIES))
        raise ValueError(f"unknown policy {name!r}; known: {known}") from None


__all__ = ["POLICIES", "SGLangCurrent", "get_policy"]

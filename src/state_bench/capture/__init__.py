"""Engine capture: record traces from live serving systems.

A source observes a system and emits :class:`CapturedRequest` values; the
:class:`TraceSink` builds, validates, and writes the trace. ``proxy_openai``
is the first source: an OpenAI-compatible recording proxy.
"""

from .base import (
    HASH_SCHEME,
    CapturedRequest,
    TokenBlockHasher,
    TraceSink,
)
from .proxy_openai import OpenAIRecorder, ProxyServer

__all__ = [
    "HASH_SCHEME",
    "CapturedRequest",
    "OpenAIRecorder",
    "ProxyServer",
    "TokenBlockHasher",
    "TraceSink",
]

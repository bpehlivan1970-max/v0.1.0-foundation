"""Runtime configuration for the Claude foundation layer.

Every setting can be overridden with an environment variable so the same
code runs unchanged in development, CI, and production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, helpful technical assistant. "
    "Prefer structured answers (steps, tables, short code examples) when they aid clarity."
)

# Server-side refusal fallback: on a safety-classifier decline the API re-runs
# the request on an Anthropic-recommended model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    # Claude Opus 5.5 defaults to "medium"; we set it explicitly so behavior
    # does not shift silently if the model's default changes.
    effort: Effort = "medium"
    max_tokens: int = 64000  # streaming, so a generous ceiling is safe
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    show_thinking: bool = False  # True -> request and print summarized thinking
    use_fallbacks: bool = True
    betas: tuple[str, ...] = field(default=(FALLBACK_BETA,))

    @classmethod
    def from_env(cls) -> "Settings":
        use_fallbacks = os.getenv("CLAUDE_USE_FALLBACKS", "1") != "0"
        return cls(
            model=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL),
            effort=os.getenv("CLAUDE_EFFORT", "medium"),  # type: ignore[arg-type]
            max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "64000")),
            system_prompt=os.getenv("CLAUDE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            show_thinking=os.getenv("CLAUDE_SHOW_THINKING", "0") == "1",
            use_fallbacks=use_fallbacks,
            betas=(FALLBACK_BETA,) if use_fallbacks else (),
        )

"""Interactive terminal chat: `python -m claude_foundation` or `claude-chat`."""

from __future__ import annotations

import sys

import anthropic

from .chat import ChatSession, RefusalError
from .config import Settings

HELP = "Commands: /reset clears history, /exit quits."


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _dim(text: str) -> None:
    _write(f"\033[2m{text}\033[0m")


def describe_error(exc: Exception) -> str:
    """Map SDK exceptions (most specific first) to an actionable message."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "Authentication failed - set ANTHROPIC_API_KEY or run `ant auth login`."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This credential lacks permission for the requested model or feature."
    if isinstance(exc, anthropic.NotFoundError):
        return f"Model or endpoint not found: {exc.message}"
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after", "?")
        return f"Rate limited (after SDK retries). Retry in {retry_after}s."
    if isinstance(exc, anthropic.BadRequestError):
        return f"Bad request: {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc.message} (request-id {exc.request_id})"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Network error - check connectivity or proxy settings."
    return f"Unexpected error: {exc!r}"


def main() -> int:
    settings = Settings.from_env()
    session = ChatSession(client=anthropic.Anthropic(), settings=settings)
    print(f"Claude chat - model {settings.model}, effort {settings.effort}. {HELP}")

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/exit":
            return 0
        if user == "/reset":
            session.reset()
            print("History cleared.")
            continue

        _write("\nclaude> ")
        try:
            result = session.send(
                user,
                on_text=_write,
                on_thinking=_dim if settings.show_thinking else None,
            )
        except RefusalError as exc:
            print(f"\n[declined: {exc.category}] {exc.explanation}")
            continue
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except anthropic.APIError as exc:
            print(f"\n[error] {describe_error(exc)}")
            continue
        except TypeError as exc:
            # The SDK raises TypeError when no credential source resolves.
            if "authentication method" not in str(exc):
                raise
            print("\n[error] No credentials found - set ANTHROPIC_API_KEY or run `ant auth login`.")
            return 1

        if result.stop_reason == "max_tokens":
            print("\n[truncated: hit max_tokens - raise CLAUDE_MAX_TOKENS]")
        footer = f"[{result.input_tokens} in / {result.output_tokens} out"
        if result.served_by != settings.model:
            footer += f", served by fallback {result.served_by}"
        _dim(f"\n{footer}]\n")


if __name__ == "__main__":
    raise SystemExit(main())

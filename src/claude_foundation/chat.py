"""Multi-turn streaming chat session on top of the Anthropic Python SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from anthropic.types.beta import BetaMessage, BetaMessageParam

from .config import Settings

TextSink = Callable[[str], None]


class RefusalError(RuntimeError):
    """Raised when the request (and any fallback model) declined to answer."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(f"Request declined (category={category}): {explanation}")


@dataclass
class TurnResult:
    text: str
    stop_reason: str | None
    served_by: str
    input_tokens: int
    output_tokens: int
    request_id: str | None


@dataclass
class ChatSession:
    """Keeps conversation history and streams each turn from Claude.

    The API is stateless, so the full history is resent every turn. We append
    the assistant's *full* content (thinking blocks included) rather than just
    its text, which keeps thinking valid across turns on the same model.
    """

    client: anthropic.Anthropic
    settings: Settings = field(default_factory=Settings.from_env)
    history: list[BetaMessageParam] = field(default_factory=list)

    def _request_kwargs(self) -> dict[str, Any]:
        thinking: dict[str, Any] = {"type": "adaptive"}
        if self.settings.show_thinking:
            thinking["display"] = "summarized"
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.settings.system_prompt,
            "thinking": thinking,
            "output_config": {"effort": self.settings.effort},
            "messages": self.history,
        }
        if self.settings.betas:
            kwargs["betas"] = list(self.settings.betas)
        if self.settings.use_fallbacks:
            kwargs["fallbacks"] = "default"
        return kwargs

    def send(
        self,
        user_message: str,
        on_text: TextSink | None = None,
        on_thinking: TextSink | None = None,
    ) -> TurnResult:
        """Send one user turn, streaming deltas to the optional callbacks."""
        self.history.append({"role": "user", "content": user_message})
        try:
            with self.client.beta.messages.stream(**self._request_kwargs()) as stream:
                for event in stream:
                    if event.type == "text" and on_text:
                        on_text(event.text)
                    elif event.type == "thinking" and on_thinking:
                        on_thinking(event.thinking)
                message = stream.get_final_message()
        except BaseException:
            # Keep history consistent: drop the user turn that never got a reply.
            self.history.pop()
            raise

        if message.stop_reason == "refusal":
            self.history.pop()
            details = message.stop_details
            raise RefusalError(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )

        self.history.append({"role": "assistant", "content": message.content})
        return _summarize(message)

    def reset(self) -> None:
        self.history.clear()


def _summarize(message: BetaMessage) -> TurnResult:
    text = "".join(b.text for b in message.content if b.type == "text")
    return TurnResult(
        text=text,
        stop_reason=message.stop_reason,
        served_by=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        request_id=getattr(message, "_request_id", None),
    )

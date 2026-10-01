# claude-foundation (v0.1.0)

A small, production-minded foundation for Claude-powered applications in Python:
an authenticated client, a streaming multi-turn chat session, and a terminal chat CLI.
Later layers (tool-using agents, document extraction, evals) can build on `ChatSession`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
claude-chat                           # or: python -m claude_foundation
```

In the chat, `/reset` clears the history and `/exit` quits.

## Using it from code

```python
import anthropic
from claude_foundation import ChatSession, RefusalError, Settings

session = ChatSession(client=anthropic.Anthropic(), settings=Settings(effort="high"))
result = session.send("Summarize Incoterms 2020 in a table.", on_text=print)
print(result.output_tokens, result.served_by)
```

## Defaults and why

| Setting | Default | Rationale |
|---|---|---|
| Model | `claude-opus-5-5` | The current Opus model. |
| Thinking | `{"type": "adaptive"}` | Claude decides how much to reason. This model can't turn thinking off. |
| Effort | `medium` (set explicitly) | Matches the model default. Raise it to `high`/`xhigh` for hard engineering work. |
| Transport | Streaming, `max_tokens=64000` | Avoids HTTP timeouts on long answers. |
| Refusal fallback | `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) | If a safety classifier declines, the API retries on a recommended model in the same call. |
| History | Full assistant `content` appended | Keeps thinking blocks valid across turns. History is append-only. |
| Retries | SDK default (2, exponential backoff) | Covers 408/409/429/5xx and network errors. |

## Configuration (environment variables)

See `.env.example`: `CLAUDE_MODEL`, `CLAUDE_EFFORT`, `CLAUDE_MAX_TOKENS`,
`CLAUDE_SYSTEM_PROMPT`, `CLAUDE_SHOW_THINKING`, `CLAUDE_USE_FALLBACKS`.

## Layout

```
src/claude_foundation/
  config.py   Settings (env-driven defaults)
  chat.py     ChatSession: streaming, history, refusal handling
  cli.py      Interactive terminal chat + error mapping
tests/        Offline tests: the real SDK running against a mock HTTP transport
```

## Tests

```bash
pytest
```

The tests make no network calls. They check the exact request shape (model, thinking,
effort, fallbacks, beta header), how history builds up over turns, and that a refusal
or API error rolls back the pending user turn.

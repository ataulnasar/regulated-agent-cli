"""Model provider abstraction.

The agent talks to one small interface, in the shape of Claude's Messages API. Swapping between the
Anthropic API, Amazon Bedrock and an offline fake is a configuration change, not a code change.
"""

import os
from dataclasses import dataclass
from typing import Any, Protocol

# Content blocks and messages are plain dicts in Messages API shape, so they can be appended to the
# conversation and replayed unchanged (including thinking blocks).
Block = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: list[Block]
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0

    def tool_uses(self) -> list[Block]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class ModelProvider(Protocol):
    name: str
    model_id: str

    def create(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ModelResponse: ...


PROVIDERS = ("fake", "anthropic", "bedrock")


def get_provider(name: str | None = None) -> ModelProvider:
    """Build the provider named by the argument or LLM_PROVIDER. Defaults to the offline fake."""
    name = (name or os.environ.get("LLM_PROVIDER") or "fake").lower()
    if name == "fake":
        from app.providers.fake import FakeProvider

        return FakeProvider()
    if name == "anthropic":
        from app.providers.claude import anthropic_provider

        return anthropic_provider()
    if name == "bedrock":
        from app.providers.claude import bedrock_provider

        return bedrock_provider()
    raise ValueError(f"unknown provider '{name}', expected one of {', '.join(PROVIDERS)}")

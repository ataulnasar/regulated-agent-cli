"""Claude via the Anthropic API or Amazon Bedrock.

Both SDK clients expose the same messages.create() surface, so one class serves both. Only the
client construction and the model ID differ.
"""

import os
from typing import Any

import anthropic

from app.providers import Message, ModelResponse

DEFAULT_MODEL_ID = "claude-sonnet-5"
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-sonnet-5"
# Stockholm. Keeps customer data and inference inside the EU for GDPR / data-residency purposes.
DEFAULT_AWS_REGION = "eu-north-1"
MAX_TOKENS = 16_000


class ClaudeProvider:
    def __init__(self, name: str, client: Any, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._client = client

    def create(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
        )
        return ModelResponse(
            content=[
                block.model_dump(mode="json", exclude_none=True) for block in response.content
            ],
            stop_reason=response.stop_reason or "unknown",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def anthropic_provider() -> ClaudeProvider:
    # Credentials resolve the SDK's usual way (ANTHROPIC_API_KEY, or an `ant auth login` profile).
    return ClaudeProvider(
        "anthropic", anthropic.Anthropic(), os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID
    )


def bedrock_provider() -> ClaudeProvider:
    # Credentials come from the standard AWS chain (env vars, profile, instance role).
    region = os.environ.get("AWS_REGION") or DEFAULT_AWS_REGION
    return ClaudeProvider(
        "bedrock",
        anthropic.AnthropicBedrockMantle(aws_region=region),
        os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_BEDROCK_MODEL_ID,
    )

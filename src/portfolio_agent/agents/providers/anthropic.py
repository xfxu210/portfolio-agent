from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from .base import JsonProvider, ProviderError, schema_instruction


class AnthropicProvider(JsonProvider):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        timeout: float = 90,
        max_retries: int = 0,
        max_output_tokens: int = 1600,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.client = Anthropic(api_key=api_key, timeout=timeout)

    def _invoke(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        request_id: str,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=schema_instruction(task, schema),
                messages=messages,
            )
        except Exception as exc:
            raise ProviderError("provider_request", str(exc), retryable=True) from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ProviderError("empty_response", "Anthropic returned no text content")
        return text, {
            "request_id": getattr(response, "id", request_id),
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }

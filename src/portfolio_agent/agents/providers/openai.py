from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import JsonProvider, ProviderError, schema_instruction


class OpenAIProvider(JsonProvider):
    provider = "openai"

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
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    def _invoke(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        request_id: str,
    ) -> tuple[str, dict[str, Any]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": schema_instruction(task, schema)},
                    *messages,
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # SDK exceptions vary by version and transport.
            raise ProviderError("provider_request", str(exc), retryable=True) from exc
        content = response.choices[0].message.content
        if not content:
            raise ProviderError("empty_response", "OpenAI returned no content")
        usage = response.usage
        return content, {
            "request_id": getattr(response, "id", request_id),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }

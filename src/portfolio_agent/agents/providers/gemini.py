from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from .base import JsonProvider, ProviderError, schema_instruction


class GeminiProvider(JsonProvider):
    provider = "gemini"

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
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    def _invoke(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        request_id: str,
    ) -> tuple[str, dict[str, Any]]:
        prompt = (
            schema_instruction(task, schema)
            + "\n"
            + "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception as exc:
            raise ProviderError("provider_request", str(exc), retryable=True) from exc
        if not response.text:
            raise ProviderError("empty_response", "Gemini returned no text content")
        metadata = getattr(response, "usage_metadata", None)
        return response.text, {
            "request_id": request_id,
            "input_tokens": getattr(metadata, "prompt_token_count", None),
            "output_tokens": getattr(metadata, "candidates_token_count", None),
        }

from __future__ import annotations

from openai import OpenAI

from .openai import OpenAIProvider


class MuseProvider(OpenAIProvider):
    provider = "muse"

    def __init__(
        self,
        model: str = "muse-spark-1.3",
        *,
        api_key: str,
        timeout: float = 90,
        max_retries: int = 0,
        max_output_tokens: int = 1600,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=api_key, base_url="https://api.meta.ai/v1", timeout=timeout)

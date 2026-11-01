from __future__ import annotations

import os
from pathlib import Path

from portfolio_agent.config import AgentConfig

from .anthropic import AnthropicProvider
from .base import LLMProvider, ProviderError
from .budgeted import BudgetedProvider
from .gemini import GeminiProvider
from .mock import MockProvider
from .muse import MuseProvider
from .openai import OpenAIProvider
from .recorded import RecordedProvider
from .routed import RoutedProvider

KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "muse": "MODEL_API_KEY",
}


def _build_live(config: AgentConfig, env: dict[str, str]) -> LLMProvider:
    assert config.provider is not None and config.model is not None
    env_name = KEYS[config.provider]
    key = env.get(env_name)
    if not key:
        raise ProviderError("missing_credentials", f"{env_name} is required for {config.provider}")
    constructors = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "muse": MuseProvider,
    }
    return constructors[config.provider](
        config.model,
        api_key=key,
        timeout=config.generation.timeout_seconds,
        max_retries=config.generation.max_retries,
        max_output_tokens=config.generation.max_output_tokens,
    )


def build_provider(
    config: AgentConfig, *, recording: Path | None = None, environment: dict[str, str] | None = None
) -> LLMProvider:
    if config.mode == "mock":
        return MockProvider()
    if config.mode == "recorded":
        if recording is None:
            raise ProviderError("recording_required", "recorded mode requires a recording path")
        return RecordedProvider(recording)
    env = dict(os.environ if environment is None else environment)
    default = _build_live(config, env)
    routes: dict[str, LLMProvider] = {}
    for stage, route in config.stage_overrides.items():
        stage_config = config.model_copy(
            update={"provider": route.provider, "model": route.model, "stage_overrides": {}}
        )
        routes[stage] = _build_live(stage_config, env)
    provider: LLMProvider = RoutedProvider(default, routes) if routes else default
    return BudgetedProvider(provider, config.budgets)

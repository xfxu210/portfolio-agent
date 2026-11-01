import json
from types import SimpleNamespace

import pytest

from portfolio_agent.agents.providers.anthropic import AnthropicProvider
from portfolio_agent.agents.providers.gemini import GeminiProvider
from portfolio_agent.agents.providers.muse import MuseProvider
from portfolio_agent.agents.providers.openai import OpenAIProvider
from portfolio_agent.domain.schemas import ExperimentProposal


def _payload() -> str:
    return json.dumps(
        {
            "schema_version": "1.0.0",
            "experiment_type": "risk_aversion",
            "hypothesis": "Use identical typed output across adapters.",
            "evidence_ids": [],
            "diagnosis_ids": [],
            "overrides": {"risk_aversion": 2.0},
            "expected_tradeoffs": ["Lower risk may reduce return."],
        }
    )


class OpenAICompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="request-openai",
            choices=[SimpleNamespace(message=SimpleNamespace(content=_payload()))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
        )


class AnthropicMessages:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="request-anthropic",
            content=[SimpleNamespace(type="text", text=_payload())],
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )


class GeminiModels:
    def __init__(self) -> None:
        self.kwargs = {}

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            text=_payload(),
            usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
        )


@pytest.mark.parametrize("provider_class", [OpenAIProvider, MuseProvider])
def test_openai_compatible_adapters_normalize_schema(provider_class) -> None:
    provider = provider_class.__new__(provider_class)
    provider.model = "explicit-model"
    provider.max_retries = 0
    provider.max_output_tokens = 321
    completions = OpenAICompletions()
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = provider.generate_structured(
        task="experiment_proposal",
        messages=[{"role": "user", "content": "{}"}],
        response_model=ExperimentProposal,
        request_id="request-local",
    )
    assert result.provider == provider_class.provider
    assert result.value.overrides == {"risk_aversion": 2.0}
    assert result.input_tokens == 12
    assert completions.kwargs["max_completion_tokens"] == 321


def test_anthropic_adapter_normalizes_schema_and_usage() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "explicit-model"
    provider.max_retries = 0
    provider.max_output_tokens = 321
    messages = AnthropicMessages()
    provider.client = SimpleNamespace(messages=messages)
    result = provider.generate_structured(
        task="experiment_proposal",
        messages=[{"role": "user", "content": "{}"}],
        response_model=ExperimentProposal,
        request_id="request-local",
    )
    assert result.provider == "anthropic"
    assert result.value.overrides == {"risk_aversion": 2.0}
    assert result.output_tokens == 8
    assert messages.kwargs["max_tokens"] == 321


def test_gemini_adapter_normalizes_schema_and_usage() -> None:
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "explicit-model"
    provider.max_retries = 0
    provider.max_output_tokens = 321
    models = GeminiModels()
    provider.client = SimpleNamespace(models=models)
    result = provider.generate_structured(
        task="experiment_proposal",
        messages=[{"role": "user", "content": "{}"}],
        response_model=ExperimentProposal,
        request_id="request-local",
    )
    assert result.provider == "gemini"
    assert result.value.overrides == {"risk_aversion": 2.0}
    assert result.request_id == "request-local"
    assert result.input_tokens == 12

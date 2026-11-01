from pathlib import Path

import pytest

from portfolio_agent.agents.providers.base import ProviderError
from portfolio_agent.agents.providers.budgeted import BudgetedProvider
from portfolio_agent.agents.providers.mock import MockProvider
from portfolio_agent.agents.providers.recorded import RecordedProvider
from portfolio_agent.agents.providers.registry import build_provider
from portfolio_agent.agents.providers.routed import RoutedProvider
from portfolio_agent.config import AgentConfig, StageRoute
from portfolio_agent.domain.schemas import ExperimentProposal


def test_mock_provider_returns_valid_domain_schema() -> None:
    provider = MockProvider()
    result = provider.generate_structured(
        task="experiment_proposal",
        messages=[{"role": "user", "content": "test"}],
        response_model=ExperimentProposal,
        request_id="request-1",
    )
    assert result.value.experiment_type == "research_blend"
    assert result.provider == "mock"


@pytest.mark.parametrize(
    ("provider", "model", "env_name"),
    [
        ("openai", "explicit-openai-model", "OPENAI_API_KEY"),
        ("anthropic", "explicit-claude-model", "ANTHROPIC_API_KEY"),
        ("gemini", "explicit-gemini-model", "GOOGLE_API_KEY"),
        ("muse", "muse-spark-1.3", "MODEL_API_KEY"),
    ],
)
def test_live_provider_missing_key_fails_before_call(provider, model, env_name) -> None:
    config = AgentConfig(mode="live", provider=provider, model=model)
    with pytest.raises(ProviderError, match=env_name):
        build_provider(config, environment={})


def test_recorded_genuine_openai_output_replays_exactly() -> None:
    provider = RecordedProvider(Path("tests/fixtures/openai_recording.jsonl"))
    result = provider.generate_structured(
        task="experiment_proposal",
        messages=[],
        response_model=ExperimentProposal,
        request_id="live-smoke-fixture",
    )
    assert result.provider == "openai"
    assert result.model == "gpt-4.1-mini"
    assert result.raw_hash == "a2f5d5823f2b19b8f7eac6c63ba6a4a0c1ed19192fbb48b74e60ea26e2564651"
    assert result.value.overrides == {"risk_aversion": 0.8}


def test_live_stage_override_builds_explicit_router() -> None:
    config = AgentConfig(
        mode="live",
        provider="openai",
        model="default-model",
        stage_overrides={
            "stock_forecast": StageRoute(provider="openai", model="forecast-model")
        },
    )
    provider = build_provider(config, environment={"OPENAI_API_KEY": "test-key"})
    assert isinstance(provider, BudgetedProvider)
    assert isinstance(provider.inner, RoutedProvider)
    assert provider.inner.default.model == "default-model"
    assert provider.inner.routes["stock_forecast"].model == "forecast-model"

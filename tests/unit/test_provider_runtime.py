import json

import pytest

from portfolio_agent.agents.providers.base import JsonProvider, ProviderError
from portfolio_agent.agents.providers.budgeted import BudgetedProvider
from portfolio_agent.agents.providers.mock import MockProvider
from portfolio_agent.config import BudgetConfig
from portfolio_agent.domain.schemas import ExperimentProposal


class FlakyJsonProvider(JsonProvider):
    provider = "test"
    model = "test-model"
    max_retries = 1
    retry_base_seconds = 0

    def __init__(self) -> None:
        self.invocations = 0

    def _invoke(self, **kwargs):
        del kwargs
        self.invocations += 1
        if self.invocations == 1:
            return "{}", {}
        return json.dumps(
            {
                "schema_version": "1.0.0",
                "experiment_type": "risk_aversion",
                "hypothesis": "Retry malformed structured output.",
                "evidence_ids": [],
                "diagnosis_ids": [],
                "overrides": {"risk_aversion": 2.0},
                "expected_tradeoffs": [],
            }
        ), {"request_id": "retry-ok"}


def test_structured_provider_retries_bounded_schema_failure() -> None:
    provider = FlakyJsonProvider()
    result = provider.generate_structured(
        task="experiment_proposal",
        messages=[],
        response_model=ExperimentProposal,
        request_id="proposal:2025-01-02:0",
    )
    assert provider.invocations == 2
    assert result.request_id == "retry-ok"


def test_budget_provider_fails_before_exceeding_monthly_calls() -> None:
    provider = BudgetedProvider(
        MockProvider(),
        BudgetConfig(
            max_model_calls_per_month=1,
            max_total_input_tokens=10,
            max_total_output_tokens=10,
        ),
    )
    provider.generate_structured(
        task="experiment_proposal",
        messages=[],
        response_model=ExperimentProposal,
        request_id="proposal:2025-01-02:0",
    )
    with pytest.raises(ProviderError, match="budget exceeded"):
        provider.generate_structured(
            task="experiment_proposal",
            messages=[],
            response_model=ExperimentProposal,
            request_id="proposal:2025-01-15:1",
        )

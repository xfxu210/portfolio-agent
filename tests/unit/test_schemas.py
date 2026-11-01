from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from portfolio_agent.domain.schemas import (
    ArticleRecord,
    ExperimentProposal,
    ExperimentType,
    ForecastReconciliation,
    Holding,
    InitialPortfolio,
    PortfolioConstraints,
)


def test_initial_portfolio_requires_exact_distinct_weights() -> None:
    holdings = tuple(Holding(symbol=f"ASSET{i:02d}", weight=0.1) for i in range(10))
    portfolio = InitialPortfolio(
        portfolio_id="initial",
        as_of=date(2024, 1, 2),
        base_currency="USD",
        benchmark="BENCH",
        initial_nav=100_000.0,
        holdings=holdings,
        constraints=PortfolioConstraints(),
        is_synthetic=True,
        data_version="test",
    )
    assert len(portfolio.holdings) == 10


def test_article_cannot_exist_before_publication() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        ArticleRecord(
            record_id="a1",
            symbols=("ASSET00",),
            title="Event",
            source_id="synthetic",
            published_at=datetime(2024, 1, 2, 15, tzinfo=UTC),
            available_at=datetime(2024, 1, 2, 14, tzinfo=UTC),
            license="synthetic-generated",
            is_synthetic=True,
            provenance="fixture",
        )


@pytest.mark.parametrize(
    ("kind", "key", "value"),
    [
        (ExperimentType.RESEARCH_BLEND, "research_blend", 1.01),
        (ExperimentType.RISK_AVERSION, "risk_aversion", 9.0),
        (ExperimentType.COVARIANCE_LOOKBACK, "covariance_lookback_sessions", 100),
    ],
)
def test_experiment_allowlist_rejects_invalid_values(kind, key, value) -> None:
    with pytest.raises(ValidationError):
        ExperimentProposal(
            experiment_type=kind,
            hypothesis="test",
            overrides={key: value},
            expected_tradeoffs=("none",),
        )


def test_forecast_reconciliation_requires_complete_contiguous_ranks() -> None:
    with pytest.raises(ValidationError, match="relative ranks"):
        ForecastReconciliation.model_validate(
            {
                "decision_as_of": datetime(2025, 1, 2, 23, 59, 59, tzinfo=UTC),
                "market_regime": "range_bound",
                "market_expected_return_3m": 0.02,
                "common_factor_assumptions": (),
                "forecasts": tuple(
                    {
                        "symbol": symbol,
                        "source_forecast_id": f"forecast-{symbol}",
                        "raw_expected_return_3m": 0.01,
                        "expected_return_3m": 0.01,
                        "confidence": 0.5,
                        "relative_rank": 2,
                        "bear_case_return": -0.05,
                        "bull_case_return": 0.08,
                        "adjustment_rationale": "test",
                    }
                    for symbol in ("A", "B")
                ),
                "cross_sectional_risks": (),
                "confidence": 0.5,
                "rationale": "test",
            }
        )

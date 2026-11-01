from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from portfolio_agent.domain.schemas import (
    ArticleRecord,
    ExperimentProposal,
    ExperimentType,
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

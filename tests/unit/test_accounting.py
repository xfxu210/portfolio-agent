import pytest

from portfolio_agent.portfolio.accounting import PortfolioAccount


def test_accounting_applies_costs_drift_and_rebalance() -> None:
    account = PortfolioAccount.from_weights(
        nav=100.0,
        weights={"A": 1.0},
        prices={"A": 10.0},
        fee_bps=100.0,
    )
    assert account.value({"A": 10.0}) == pytest.approx(99.0)
    assert account.value({"A": 20.0}) == pytest.approx(198.0)
    turnover, fees = account.rebalance(
        {"A": 0.5, "B": 0.5},
        {"A": 20.0, "B": 10.0},
    )
    assert turnover == pytest.approx(0.5)
    assert fees == pytest.approx(0.99)
    assert account.value({"A": 20.0, "B": 10.0}) == pytest.approx(197.01)
    assert account.weights({"A": 22.0, "B": 10.0})["A"] > 0.5

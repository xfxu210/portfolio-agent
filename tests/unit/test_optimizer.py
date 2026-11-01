import numpy as np
import pandas as pd

from portfolio_agent.portfolio.optimizer import optimize_portfolio


def test_optimizer_respects_hard_constraints() -> None:
    symbols = [f"ASSET{i:02d}" for i in range(10)]
    expected = pd.Series(np.linspace(0.01, 0.08, 10), index=symbols)
    covariance = pd.DataFrame(np.eye(10) * 0.02, index=symbols, columns=symbols)
    previous = {symbol: 0.1 for symbol in symbols}
    result = optimize_portfolio(expected, covariance, previous)
    assert result.status == "optimal"
    assert abs(sum(result.weights.values()) - 1) < 1e-6
    assert all(0.02 - 1e-6 <= value <= 0.25 + 1e-6 for value in result.weights.values())
    assert result.one_way_turnover <= 0.20 + 1e-6


def test_optimizer_rejects_wrong_holding_count() -> None:
    symbols = [f"ASSET{i:02d}" for i in range(9)]
    expected = pd.Series(0.05, index=symbols)
    covariance = pd.DataFrame(np.eye(9), index=symbols, columns=symbols)
    result = optimize_portfolio(expected, covariance, {})
    assert result.status == "infeasible"

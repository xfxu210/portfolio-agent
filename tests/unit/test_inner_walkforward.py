import numpy as np
import pandas as pd

from portfolio_agent.evaluation.inner_walkforward import trailing_walkforward_score


def test_inner_walkforward_uses_trailing_chronological_rows_only() -> None:
    index = pd.date_range("2024-01-01", periods=126, freq="B")
    returns = pd.DataFrame(
        {"A": np.full(126, 0.001), "B": np.full(126, -0.0005)},
        index=index,
    )
    weights = pd.Series({"A": 0.75, "B": 0.25})
    score = trailing_walkforward_score(
        returns,
        weights,
        risk_aversion=2.0,
        one_way_turnover=0.1,
        fee_bps=10,
        calibration_error=0.02,
        confidence_scale=0.5,
    )
    assert score > 0
    future = pd.DataFrame({"A": [-1.0], "B": [-1.0]}, index=[pd.Timestamp("2030-01-01")])
    unchanged = trailing_walkforward_score(
        returns,
        weights,
        risk_aversion=2.0,
        one_way_turnover=0.1,
        fee_bps=10,
        calibration_error=0.02,
        confidence_scale=0.5,
    )
    assert unchanged == score
    assert future.index.min() > returns.index.max()

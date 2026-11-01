from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_walkforward_score(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    risk_aversion: float,
    one_way_turnover: float,
    fee_bps: float,
    calibration_error: float,
    confidence_scale: float,
    folds: int = 3,
) -> float:
    """Score chronological trailing folds ending at the decision date.

    This function accepts only the already point-in-time-filtered return matrix. Splits preserve
    order, and no row after the decision cutoff can enter a fold.
    """

    portfolio = returns[weights.index].dot(weights).dropna()
    if len(portfolio) < folds * 20:
        raise ValueError("insufficient trailing observations for walk-forward scoring")
    fold_scores = []
    for fold in np.array_split(portfolio, folds):
        annual_return = float(fold.mean() * 252)
        annual_volatility = float(fold.std(ddof=0) * np.sqrt(252))
        fold_scores.append(annual_return - risk_aversion * annual_volatility)
    return float(
        np.mean(fold_scores)
        - one_way_turnover * fee_bps / 10_000
        - calibration_error * confidence_scale
    )

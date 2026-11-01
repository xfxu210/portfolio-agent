from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(nav: pd.Series) -> dict[str, float]:
    returns = nav.pct_change().dropna()
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1)
    volatility = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)) if volatility > 0 else 0.0
    drawdown = nav / nav.cummax() - 1
    return {
        "total_return": total_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }

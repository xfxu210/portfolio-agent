from __future__ import annotations

from datetime import date

import numpy as np


def projection_fan(
    *,
    initial_nav: float,
    expected_return_3m: float,
    annualized_volatility: float,
    future_sessions: list[date],
    seed: int,
    simulations: int = 2000,
) -> dict[str, list[tuple[date, float]]]:
    horizon = len(future_sessions)
    if horizon != 63:
        raise ValueError("projection requires exactly 63 future sessions")
    rng = np.random.default_rng(seed)
    daily_mu = np.log1p(expected_return_3m) / horizon
    daily_sigma = annualized_volatility / np.sqrt(252)
    shocks = rng.normal(daily_mu - 0.5 * daily_sigma**2, daily_sigma, (simulations, horizon))
    paths = initial_nav * np.exp(np.cumsum(shocks, axis=1))
    quantiles = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    return {
        name: [
            (day, round(float(value), 6))
            for day, value in zip(future_sessions, values, strict=True)
        ]
        for name, values in zip(("p10", "p50", "p90"), quantiles, strict=True)
    }

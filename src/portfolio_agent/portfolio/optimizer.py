from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    weights: dict[str, float]
    one_way_turnover: float
    objective_value: float | None
    reason: str | None = None


def select_ten(signals: pd.DataFrame, previous: dict[str, float]) -> list[str]:
    ranked = signals["symbol"].tolist()
    incumbent = [symbol for symbol in previous if symbol in set(ranked)]
    if len(incumbent) < 10:
        incumbent.extend(symbol for symbol in ranked if symbol not in incumbent)
        return incumbent[:10]
    incumbent_ranked = [symbol for symbol in ranked if symbol in incumbent]
    challengers = [symbol for symbol in ranked if symbol not in incumbent]
    selected = incumbent_ranked[:]
    replacement_budget = 0.18  # leave numerical room for reweighting inside the 20% cap
    replaced_weight = 0.0
    for challenger, incumbent_symbol in zip(challengers, reversed(incumbent_ranked), strict=False):
        exit_weight = previous.get(incumbent_symbol, 0.0)
        if replaced_weight + exit_weight > replacement_budget:
            continue
        selected.remove(incumbent_symbol)
        selected.append(challenger)
        replaced_weight += exit_weight
        selected = sorted(selected, key=ranked.index)
    return selected[:10]


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    previous: dict[str, float],
    *,
    risk_aversion: float = 2.0,
    turnover_penalty: float = 0.5,
    max_turnover: float = 0.20,
    min_weight: float = 0.02,
    max_weight: float = 0.25,
) -> OptimizationResult:
    symbols = expected_returns.index.tolist()
    if len(symbols) != 10 or len(set(symbols)) != 10:
        return OptimizationResult("infeasible", {}, 0.0, None, "exactly ten symbols required")
    cov = covariance.reindex(index=symbols, columns=symbols).to_numpy(dtype=float)
    cov = (cov + cov.T) / 2 + np.eye(len(symbols)) * 1e-8
    mu = expected_returns.to_numpy(dtype=float)
    prev = np.array([previous.get(symbol, 0.0) for symbol in symbols])
    exited = sum(weight for symbol, weight in previous.items() if symbol not in symbols)
    weights = cp.Variable(len(symbols))
    one_way_turnover = 0.5 * (cp.norm1(weights - prev) + exited)
    objective = cp.Maximize(
        mu @ weights
        - risk_aversion * cp.quad_form(weights, cp.psd_wrap(cov))
        - turnover_penalty * one_way_turnover
    )
    constraints = [
        cp.sum(weights) == 1,
        weights >= min_weight,
        weights <= max_weight,
        one_way_turnover <= max_turnover + 1e-9,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:
        return OptimizationResult("error", {}, 0.0, None, str(exc))
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        return OptimizationResult("infeasible", {}, 0.0, None, problem.status)
    result = {
        symbol: max(0.0, float(value)) for symbol, value in zip(symbols, weights.value, strict=True)
    }
    total = sum(result.values())
    result = {symbol: value / total for symbol, value in result.items()}
    turnover = 0.5 * (
        sum(
            abs(result.get(symbol, 0.0) - previous.get(symbol, 0.0))
            for symbol in set(result) | set(previous)
        )
    )
    return OptimizationResult("optimal", result, turnover, float(problem.value))

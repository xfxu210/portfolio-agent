from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from portfolio_agent.data.loader import DatasetStore


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def build_signal_table(
    store: DatasetStore,
    day: date,
    *,
    research_blend: float,
    lookback: int = 126,
) -> pd.DataFrame:
    symbols = store.eligible_symbols(day)
    history = store.price_history(day, symbols)
    pivot = history.pivot(index="date", columns="symbol", values="adjusted_close").tail(
        lookback + 1
    )
    if len(pivot) < lookback + 1:
        raise ValueError(f"insufficient history for {lookback}-session signals")
    momentum = pivot.iloc[-1] / pivot.iloc[0] - 1
    volatility = pivot.pct_change().iloc[1:].std(ddof=0) * np.sqrt(252)
    quant = _zscore(momentum) - 0.25 * _zscore(volatility)

    recent = store.research_as_of(day, symbols)
    recent = recent[recent["published_at"].dt.date >= day - timedelta(days=90)].copy()
    scores = {symbol: 0.0 for symbol in symbols}
    counts = {symbol: 0 for symbol in symbols}
    seen_events: set[str] = set()
    for row in recent.itertuples():
        event_key = row.event_id or row.record_id
        if event_key in seen_events:
            continue
        seen_events.add(event_key)
        text = f"{row.title} {row.summary or ''}".lower()
        value = float("positive" in text) - float("negative" in text)
        age = max(0, (day - row.published_at.date()).days)
        value *= float(np.exp(-age / 45))
        for symbol in row.symbols:
            if symbol in scores:
                scores[symbol] += value
                counts[symbol] += 1
    research = pd.Series(scores, dtype=float)
    coverage = pd.Series(counts, dtype=int)
    research_standard = _zscore(research)
    no_research = coverage == 0
    research_standard.loc[no_research] = 0.0
    rank = research_blend * research_standard + (1 - research_blend) * quant
    return pd.DataFrame(
        {
            "symbol": symbols,
            "momentum": momentum.reindex(symbols).values,
            "annualized_volatility": volatility.reindex(symbols).values,
            "quant_score": quant.reindex(symbols).values,
            "research_score": research_standard.reindex(symbols).values,
            "research_count": coverage.reindex(symbols).values,
            "rank_score": rank.reindex(symbols).values,
        }
    ).sort_values(["rank_score", "symbol"], ascending=[False, True])

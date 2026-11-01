from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class PortfolioAccount:
    cash: float
    shares: dict[str, float]
    fee_bps: float

    @classmethod
    def from_weights(
        cls, nav: float, weights: dict[str, float], prices: dict[str, float], fee_bps: float
    ) -> PortfolioAccount:
        turnover = sum(weights.values())
        fees = nav * turnover * fee_bps / 10_000
        investable = nav - fees
        shares = {
            symbol: investable * weight / prices[symbol] for symbol, weight in weights.items()
        }
        return cls(cash=0.0, shares=shares, fee_bps=fee_bps)

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(
            self.shares.get(symbol, 0.0) * price for symbol, price in prices.items()
        )

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        nav = self.value(prices)
        return {symbol: self.shares[symbol] * prices[symbol] / nav for symbol in self.shares}

    def rebalance(self, target: dict[str, float], prices: dict[str, float]) -> tuple[float, float]:
        nav = self.value(prices)
        current = self.weights(prices)
        turnover = 0.5 * sum(
            abs(target.get(symbol, 0.0) - current.get(symbol, 0.0))
            for symbol in set(target) | set(current)
        )
        fees = nav * turnover * self.fee_bps / 10_000
        investable = nav - fees
        self.shares = {
            symbol: investable * weight / prices[symbol] for symbol, weight in target.items()
        }
        self.cash = 0.0
        return turnover, fees


def price_map(prices: pd.DataFrame, day: date) -> dict[str, float]:
    frame = prices[prices["date"] == day]
    return dict(zip(frame["symbol"], frame["adjusted_close"], strict=True))

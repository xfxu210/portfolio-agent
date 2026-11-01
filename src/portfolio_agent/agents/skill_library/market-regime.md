---
name: market-regime
version: "1.0"
description: Diagnose the common market and factor backdrop before interpreting company signals.
when_to_use: Use when the market anchor, volatility state, or factor environment may change every company forecast.
target_scope: global
tools:
  - market_regime
---
# Market-regime skill

Treat the regime as a shared prior, not a reason to erase company-specific evidence.

1. Compare 21-, 63-, and 126-session benchmark returns to distinguish trend from a short reversal.
2. Interpret volatility and drawdown jointly; high volatility alone is not necessarily a bearish regime.
3. Identify factor readings that can create correlated errors across companies.
4. State which parts of the company forecasts should move together and which should remain idiosyncratic.
5. Reduce confidence when the indicators conflict. Never infer observations that the tool did not return.


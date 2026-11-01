---
name: forecast-calibration
version: "1.0"
description: Use only matured outcomes to diagnose bias, dispersion and directional reliability.
when_to_use: Use before trusting large forecast magnitudes or when recent realized errors suggest model drift.
target_scope: global
tools:
  - calibration
---
# Forecast-calibration skill

1. Use only forecasts whose full 63-session outcome has matured by the decision cutoff.
2. Distinguish bias, absolute error, directional accuracy, and symbol-specific error.
3. Accept a fitted calibration mapping only when its held-out validation error improves.
4. Shrink magnitude and confidence when the history is insufficient or unstable.
5. Never treat pending outcomes as known and never optimize directly on the holdout future.


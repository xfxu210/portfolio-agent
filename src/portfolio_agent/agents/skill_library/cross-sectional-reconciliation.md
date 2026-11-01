---
name: cross-sectional-reconciliation
version: "1.0"
description: Reconcile deterministic priors and LLM forecasts into one coherent all-company distribution.
when_to_use: Mandatory final skill before forecasts are allowed to reach portfolio construction.
target_scope: cross_section
tools: []
---
# Cross-sectional reconciliation skill

For every eligible company exactly once:

1. Keep the deterministic point estimate, standalone LLM estimate, revisions, and tool evidence distinct.
2. Explain material movement away from the deterministic prior and raw LLM forecast.
3. Remove duplicated market or sector effects and correct unsupported peer dispersion.
4. Apply only calibration learned from matured outcomes.
5. Make confidence reflect evidence agreement, data coverage, and historical reliability.
6. Produce contiguous relative ranks plus coherent bear, base, and bull scenarios.
7. When evidence is weak or contradictory, shrink toward the deterministic inference rather than forcing conviction.


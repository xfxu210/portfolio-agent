# Methodology

## Chronology

At each monthly decision close, the engine retrieves only records whose publication and
availability timestamps are no later than the cutoff. It executes accepted weights at the
next session close and begins earning returns afterward. All compared strategies use that
same approximation. Transaction costs are intentionally ignored in the current forecasting-focused
iteration.

```text
decision close → dynamic research actions → all-company reconciliation
               → freeze forecast and portfolio → next-session close execution
               → portfolio drift → 63-session outcome maturity
```

## Dynamic forecasting agent

Every eligible company first receives a deterministic inference built from all available data
modalities and a dated, evidence-grounded standalone LLM analysis and return forecast. The model
then enters a bounded decide-act-observe loop. It selects load-on-demand skills for market regime,
company deep dives, peer relative value, calibration, or portfolio awareness. Each skill has a
versioned playbook and an explicit tool allowlist. The skill sequence is chosen at runtime from the
current state; it is not a hard-coded sequence in live mode.

Before finishing, the model must load the mandatory reconciliation playbook and jointly reconcile
the deterministic inference, standalone LLM estimate, revisions, and tool evidence for every
eligible company. The resulting
cross-section contains one source-linked estimate per company, contiguous relative ranks, bear and
bull cases, calibrated confidence, common-factor assumptions, and explicit adjustment rationale.
Unknown companies, missing forecasts, mismatched forecast IDs, duplicate ranks, and mismatched
decision dates fail the run.

After 63 complete sessions, individual company forecasts are scored for absolute error, bias, and
direction. Those outcomes are unavailable before maturity and become calibration observations for
later decisions. The run reports standalone-versus-reconciled MAE and directional accuracy so that
reconciliation is measured rather than assumed to help.

## Nested experimentation

After forecast reconciliation, the within-month strategy loop proposes up to three one-family
parameter changes, solves each portfolio deterministically, evaluates it using trailing returns
only, and freezes its projection before choosing a winner. The across-month loop exposes completed
63-session forecast errors to later decisions. Pending outcomes are never treated as known.

The development interval selects the fixed comparator and freezes the meta-policy before the
holdout interval. Holdout decisions may react only to observations that have become available
inside the chronology.

## Portfolio rules

- Exactly ten equities, long only, no leverage
- 2–25% per holding
- 20% maximum one-way discretionary turnover
- Transaction fees set to zero while forecast quality is the active research objective
- Fixed formation cohort with dated eligibility
- Rejected or infeasible experiments remain in the audit log

## Projections

Every feasible trial receives a seeded Monte Carlo fan with p10, p50, and p90 dated paths.
The fan uses the candidate expected return and covariance-derived annualized volatility.
It is a forecast artifact, never a realized curve.

# Methodology

## Chronology

At each monthly decision close, the engine retrieves only records whose publication and
availability timestamps are no later than the cutoff. It executes accepted weights at the
next session close and begins earning returns afterward. All compared strategies use that
same approximation and fee schedule.

```text
decision close → freeze evidence and proposal → next-session close execution
               → portfolio drift and costs → 63-session outcome maturity
```

## Nested experimentation

The within-month loop analyzes a shortlist once, proposes up to three one-family parameter
changes, solves each portfolio deterministically, evaluates it using trailing returns only,
and freezes its projection before choosing a winner. The across-month loop exposes completed
63-session forecast errors to later proposals. Pending outcomes are never treated as known.

The development interval selects the fixed comparator and freezes the meta-policy before the
holdout interval. Holdout decisions may react only to observations that have become available
inside the chronology.

## Portfolio rules

- Exactly ten equities, long only, no leverage
- 2–25% per holding
- 20% maximum one-way discretionary turnover
- Fees charged on one-way turnover
- Fixed formation cohort with dated eligibility
- Rejected or infeasible experiments remain in the audit log

## Projections

Every feasible trial receives a seeded Monte Carlo fan with p10, p50, and p90 dated paths.
The fan uses the candidate expected return and covariance-derived annualized volatility.
It is a forecast artifact, never a realized curve.


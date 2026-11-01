# Architecture

```text
Validated immutable dataset
          │
          ▼
 Point-in-time store ───────────── matured company forecast outcomes
          │                                      │
          └────────────────┬─────────────────────┘
                           ▼
                 Forecasting agent loop
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       choose action   execute tool   inspect observation
            ▲                              │
            └──────── revise / continue ───┘
                           │
                           ▼
          mandatory all-company reconciliation
                           │
                           ▼
              deterministic portfolio engine
        selection / constraints / optimization / accounting
                           │
                           ▼
                append-only run artifacts
                           │
                           ▼
                   FastAPI + dashboard
```

The model owns the forecasting research policy. At each monthly cutoff it sees the current
forecast state and decides which bounded action has the highest expected information gain: inspect
the market regime, compare sector peers, inspect company history, review matured calibration,
revisit selected companies, reconcile the cross-section, or finish. Actions produce observations
that become input to the next decision. Duplicate actions, unknown symbols, call budgets, and step
budgets are enforced by code.

The agent must reconcile every eligible company's standalone forecast before its output can reach
portfolio construction. Reconciliation corrects common-factor double counting, relative-value
inconsistency, implausible dispersion, and confidence calibration. Company forecast errors mature
after 63 sessions and become available to later agent decisions.

The LLM cannot execute code, submit arbitrary weights, see future observations, alter the horizon,
or bypass portfolio constraints. Deterministic code executes tools, validates provenance, and
constructs the final ten-stock portfolio from the reconciled forecast set. JSONL and Parquet remain
the portable run contract; the dashboard is a read-only view of those artifacts.

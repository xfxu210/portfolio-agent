# Architecture

```text
Validated immutable dataset
          │
          ▼
 Point-in-time store ───────────── matured company forecast outcomes
          │                                      │
          └────────────────┬─────────────────────┘
                           ▼
         deterministic all-data inference layer
                           │
                           ▼
                 Forecasting agent loop
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       choose skill    execute tools  inspect observation
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

Before the model acts, deterministic tools consume every point-in-time data modality: universe,
prices, benchmark, factors, dated articles, reported fundamentals, and portfolio state. They create
stable company priors, coverage diagnostics, and a market anchor. The same inputs always produce
the same inference.

The model owns the dynamic forecasting research policy. At each monthly cutoff it sees the stable
inference, current forecasts, a catalog of versioned skills, and prior observations. It chooses the
skill or bounded action with the highest expected information gain. Full skill instructions load
only when selected, and code executes only the tools allowlisted by that skill. Duplicate actions,
unknown skills or symbols, call budgets, and step budgets are enforced by code.

The agent must reconcile every eligible company's standalone forecast before its output can reach
portfolio construction. Reconciliation corrects common-factor double counting, relative-value
inconsistency, implausible dispersion, and confidence calibration. Company forecast errors mature
after 63 sessions and become available to later agent decisions.

The LLM cannot execute code, submit arbitrary weights, see future observations, alter the horizon,
or bypass portfolio constraints. Deterministic code executes tools, validates provenance, and
constructs the final ten-stock portfolio from the reconciled forecast set. JSONL and Parquet remain
the portable run contract; the dashboard is a read-only view of those artifacts.

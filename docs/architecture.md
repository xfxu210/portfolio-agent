# Architecture

```text
Validated immutable dataset
          │
          ▼
 Point-in-time store ──────── matured 63-session outcomes
          │                              │
          └──────────────┬───────────────┘
                         ▼
                Monthly controller
          ┌──────────────┴───────────────┐
          ▼                              ▼
  Provider-neutral LLM            Deterministic engine
 analysis / forecast / plan   signals / optimizer / accounting
          └──────────────┬───────────────┘
                         ▼
             Append-only run artifacts
                         ▼
                 FastAPI + dashboard
```

The LLM boundary accepts evidence and returns typed domain objects. It cannot execute code,
submit arbitrary weights, alter the horizon, or bypass constraints. JSONL and Parquet are the
portable run contract; the dashboard is a read-only view of those artifacts.

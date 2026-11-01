# Portfolio Agent

Portfolio Agent is an auditable, point-in-time portfolio research and optimization agent.
It replays monthly decisions over an immutable offline dataset, asks a selected language
model for grounded forecasts and bounded strategy experiments, and reserves ranking,
optimization, accounting, projections, and acceptance decisions for deterministic code.

The repository is independently runnable. It does not require a private optimizer or a
runtime financial-data service.

> Historical, synthetic, and retrospective results are research demonstrations—not
> investment advice, brokerage execution, or evidence of future returns.

## What the demo proves

- A fixed formation universe is filtered by historical eligibility at every decision.
- Research and fundamentals require `available_at <= decision_as_of`.
- Exactly ten long-only holdings are optimized under weight and turnover constraints.
- Each meaningful experiment receives an immutable 63-session projection fan.
- Forecast errors mature only after the full horizon and become feedback for later months.
- Original, development-selected fixed, and adaptive strategies share execution timing,
  costs, and the same calendar.
- Negative and rejected experiments remain visible.
- Mock, live, and recorded modes use the same typed provider boundary.

## Quick start

Requires Python 3.12 and Node.js 22 or newer.

```bash
make install
make smoke
make replay
make test
```

Start the local API and dashboard in separate terminals:

```bash
make api
make frontend
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000). The browser never receives provider
API keys.

## Complete synthetic replay

The independent `portfolio-agent-data-prep` workspace produces the approved
`synthetic-full-v1` package. Once generated:

```bash
portfolio-agent data validate --dataset /path/to/output/approved
portfolio-agent replay --dataset /path/to/output/approved \
  --mode mock --adaptive --rebalance monthly --iterations 3 --out runs/demo
portfolio-agent verify runs/demo
```

The demo run generates 25 monthly decisions, three realized curves, all trial projections,
matured outcomes, immutable model artifacts, a SQLite audit mirror, and a portable report.
Every artifact is hashed in `manifest.json`; `verify` recomputes the metrics and checks those
hashes. `traces.jsonl` includes normalized outputs so the whole run can be replayed offline:

```bash
portfolio-agent replay --dataset /path/to/output/approved \
  --mode recorded --recording runs/demo/traces.jsonl --out runs/recorded
portfolio-agent verify runs/recorded
```

## Live providers

Choose the provider and exact model explicitly. No cross-provider fallback occurs.

```bash
# Muse / Meta Model API
export MODEL_API_KEY=... # never commit this value
portfolio-agent replay --dataset /path/to/data --mode live \
  --provider muse --model muse-spark-1.3 --adaptive --iterations 3 --out runs/muse

# OpenAI
export OPENAI_API_KEY=...
portfolio-agent replay --dataset /path/to/data --mode live \
  --provider openai --model YOUR_MODEL_ID --adaptive --iterations 3 --out runs/openai
```

Claude uses `ANTHROPIC_API_KEY`; Gemini uses `GOOGLE_API_KEY`. See
[model provider setup](docs/model-providers.md).

The dashboard exposes the same four-provider selection and submits runs to a local in-process
queue. Credentials remain backend-only. Per-stage routes, bounded exponential retries, call and
token budgets, and strict structured-output validation apply before deterministic portfolio math.

## Project boundaries

The public runtime consumes a validated snapshot and never extracts private financial data.
The separate private preparation workspace owns source inventory, rights decisions,
normalization, synthetic-full generation, validation, and optional Hugging Face publication.
See [methodology](docs/methodology.md), [dataset contract](docs/dataset-contract.md),
[licensing](docs/data-licensing.md), and [limitations](docs/limitations.md).

## Verification

```bash
ruff check .
pytest
cd frontend && npm run lint && npm run build
```

`portfolio-agent verify RUN_DIR` recomputes every reported curve metric from stored numeric
artifacts and reports tampering or inconsistencies.

## Delivered local reference run

The checked implementation has been exercised against `synthetic-full-v1`: 756 exchange
sessions, 40 equities, 782 dated research records, 480 point-in-time fundamentals, 25 monthly
decisions, 75 experiments/projections, and 66 matured projection outcomes. The result is
deliberately negative: the adaptive run underperformed the original baseline. The UI keeps that
visible; the project makes no alpha claim.

An authenticated OpenAI `gpt-4.1-mini` structured call was also executed using an existing local
credential without copying it into this project. A secret-free normalized response is committed
as a recorded-provider contract fixture. Muse, Anthropic, and Gemini live calls remain conditional
on their respective credentials; all adapters otherwise share the tested provider contract.

# Portfolio Agent

Public repository: [github.com/xfxu210/portfolio-agent](https://github.com/xfxu210/portfolio-agent)

Portfolio Agent is an auditable, point-in-time portfolio research and optimization agent.
It replays monthly decisions over an immutable offline dataset. A selected language model runs a
bounded decide-act-observe loop, loads specialized forecasting skills, chooses which tools and
questions to pursue, revises uncertain company estimates, and reconciles the complete
eligible-company forecast cross-section. A deterministic evidence layer consumes every available
data modality before the dynamic model acts.
Portfolio constraints, optimization, accounting, projections, and artifact validation remain
deterministic.

The repository is independently runnable. It does not require a private optimizer or a
runtime financial-data service.

> Historical, synthetic, and retrospective results are research demonstrations—not
> investment advice, brokerage execution, or evidence of future returns.

## What the demo proves

- A fixed formation universe is filtered by historical eligibility at every decision.
- Research and fundamentals require `available_at <= decision_as_of`.
- A stable evidence layer combines point-in-time prices, benchmark, factors, news, fundamentals,
  universe metadata, and portfolio state into reproducible company inferences.
- The forecasting model dynamically loads versioned market, company, peer, calibration, and
  portfolio skills and executes only their allowlisted tools under explicit budgets.
- Every eligible company forecast is reconciled against all others before portfolio construction.
- Standalone and reconciled forecast accuracy are measured after the full 63-session horizon.
- Exactly ten long-only holdings are optimized under weight and turnover constraints.
- Each meaningful experiment receives an immutable 63-session projection fan.
- Forecast errors mature only after the full horizon and become feedback for later months.
- Original, development-selected fixed, and adaptive strategies share execution timing and the
  same calendar. Transaction costs are intentionally ignored in this forecasting-focused version.
- Negative and rejected experiments remain visible.
- Mock, live, and recorded modes use the same typed provider boundary.

## Quick start

Requires Python 3.12 and Node.js 22 or newer.

```bash
git clone https://github.com/xfxu210/portfolio-agent.git
cd portfolio-agent
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

After the dataset is published, download it only by its full immutable Hugging Face commit:

```bash
portfolio-agent data download \
  --repo-id <HF_NAMESPACE>/portfolio-agent-data \
  --revision <FULL_40_CHARACTER_COMMIT_SHA> \
  --output data/synthetic-full-v1
portfolio-agent data validate --dataset data/synthetic-full-v1
```

The download records the repository ID and revision in local source metadata. Validation displays
both values, and every replay copies them into its summary and run manifest. Mutable names such as
`main` are rejected.

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
[agent skills](docs/agent-skills.md), [architecture](docs/architecture.md),
[model providers](docs/model-providers.md), [licensing](docs/data-licensing.md),
[limitations](docs/limitations.md), [delivery status](docs/delivery-status.md), and the
[demo script](docs/demo-script.md).

## Verification

```bash
ruff check .
pytest
cd frontend && npm run lint && npm run build
```

`portfolio-agent verify RUN_DIR` recomputes every reported curve metric from stored numeric
artifacts and reports tampering or inconsistencies.

Optional robustness sweeps retain every result across projection-seed assumptions:

```bash
.venv/bin/python scripts/run_robustness.py --dataset examples/smoke \
  --out runs/robustness --seeds 42 43
```

## Delivered local reference run

The checked implementation has been exercised against `synthetic-full-v1`: 756 exchange
sessions, 40 equities, 782 dated research records, 480 point-in-time fundamentals, 25 monthly
decisions, 25 complete 40-company reconciliations, 880 matured company forecasts, and 75 strategy
experiments/projections. The deterministic mock result remains negative and does not show reliable
forecast improvement. The UI keeps standalone-versus-reconciled accuracy and portfolio failures
visible; the project makes no alpha claim.

An authenticated two-month OpenAI `gpt-4.1-mini` acceptance replay was executed using an existing
local credential without copying it into this project: 61 genuine calls produced 24 initial
company analyses, 24 forecasts, different runtime-chosen research actions in each month, two full
12-company reconciliations, and two strategy experiments. Its captured responses replayed without
a key and reproduced the same strategy and metrics exactly. A smaller secret-free response is
committed as a provider contract fixture. Muse, Anthropic, and Gemini live calls remain conditional
on their respective credentials; all adapters otherwise share the tested provider contract.

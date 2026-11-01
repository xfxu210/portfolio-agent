# Dataset Contract

Schema version `1.0.0` uses Parquet tables plus canonical JSON. Required inputs are:

- `manifest.json` with payload SHA-256 hashes
- `universe/universe.parquet`
- `market/prices.parquet`, `benchmark.parquet`, and `factors.parquet`
- partitioned `research/articles/year=YYYY/month=MM/*.parquet`
- `research/fundamentals.parquet`
- `portfolio/initial_portfolio.json`
- `reports/rights_summary.json`

All timestamps are UTC. `published_at` describes when content was published;
`available_at` describes when the exact record could first have been known. Fiscal period end
is never treated as fundamental availability. Unknown rights fail closed.

Generated analyses, forecasts, weights, experiments, projections, and realized labels are run
outputs—not source-data inputs.

The consumer requires a full immutable Hugging Face revision and independently verifies every
manifest hash after download. `portfolio-agent data download` writes the separately resolved
dataset repository ID and full 40-character commit SHA to `.portfolio-agent-source.json`; this
local source record is not part of the dataset payload manifest. Validation exposes the source,
and replay copies it into `summary.json` and the run `manifest.json`. Locally prepared datasets
honestly record both values as `null`.

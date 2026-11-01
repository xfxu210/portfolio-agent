# Delivery Status

This document separates implemented local deliverables from external publication prerequisites.

## Implemented and verified locally

- Strict point-in-time schemas, checksummed Parquet/JSON loading, and a pinned-revision-only
  Hugging Face downloader.
- Fixed formation universe, dynamic exactly-ten-stock selection, bounded CVXPY allocation,
  next-session execution, turnover costs, drift, and objective performance metrics.
- Three nested-comparison curves: original baseline, development-selected fixed strategy, and
  adaptive monthly strategy, with a sealed holdout label and no success requirement.
- Three bounded experiments per month, a frozen 63-session projection per feasible experiment,
  delayed outcomes, matured feedback, and visibly changing strategy versions.
- OpenAI, Anthropic, Gemini, and Muse adapters plus deterministic mock, exact recorded replay,
  stage routing, retries, budgets, schema validation, evidence grounding, and provenance.
- JSONL/Parquet artifacts, SQLite audit mirror, environment/model/data manifest, SHA-256 artifact
  integrity verification, FastAPI endpoints, local run queue, and Next.js dashboard.
- Full synthetic dataset and a 25-month offline run; a second recorded run reproduces the same
  decisions and metrics.

## External gates intentionally not fabricated

- Hugging Face upload requires the user's namespace, `HF_TOKEN`, and explicit publication action.
  The payload passes the dry-run gate, but no public URL or commit SHA is claimed.
- GitHub publication and demo-video upload require the destination account/repository. The two
  local repositories are complete and can be pushed without generated data, environments, keys,
  or run directories.
- Muse, Claude, and Gemini live smoke calls require `MODEL_API_KEY`, `ANTHROPIC_API_KEY`, and
  `GOOGLE_API_KEY`. Missing credentials fail before a run starts. OpenAI authentication was tested.
- No rights-cleared real historical corpus was available. The shipped three-year profile is fully
  synthetic and labeled on every record and in the dashboard.

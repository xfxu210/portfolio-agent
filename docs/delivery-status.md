# Delivery Status

This document separates implemented and published deliverables from remaining external
publication prerequisites.

## Published

- The application source, tests, smoke fixture, CI workflow, and documentation are public at
  [github.com/xfxu210/portfolio-agent](https://github.com/xfxu210/portfolio-agent).
- The CC BY 4.0 synthetic-full dataset is public at
  [cognitmarket/portfolio-agent-data](https://huggingface.co/datasets/cognitmarket/portfolio-agent-data),
  pinned at commit `74c1790faeefc527c7fd69469b0f4b50b67d3425`.

## Implemented and verified locally

- Strict point-in-time schemas, checksummed Parquet/JSON loading, and a pinned-revision-only
  Hugging Face downloader. Downloaded repository and revision metadata flow into validation,
  summaries, and run manifests.
- Fixed formation universe, dynamic exactly-ten-stock selection, bounded CVXPY allocation,
  next-session execution, drift, and objective performance metrics. Transaction costs are zero by
  explicit request in the current forecasting-focused version.
- A bounded model-controlled decide-act-observe loop with point-in-time market, peer, company
  history, and calibration tools; selective forecast revision; duplicate and budget stopping rules;
  and a mandatory source-linked reconciliation across every eligible company.
- Delayed company-level forecast maturation plus standalone-versus-reconciled MAE and directional
  accuracy, so forecast improvement is evaluated directly.
- Three nested-comparison curves: original baseline, development-selected fixed strategy, and
  adaptive monthly strategy, with a sealed holdout label and no success requirement.
- Three bounded strategy experiments per month after the dynamic forecasting loop, a frozen
  63-session projection per feasible experiment, delayed outcomes, matured feedback, and visibly
  changing strategy versions.
- OpenAI, Anthropic, Gemini, and Muse adapters plus deterministic mock, exact recorded replay,
  stage routing, retries, budgets, schema validation, evidence grounding, and provenance.
- Agent action, tool observation, reconciliation, and company-outcome artifacts; JSONL/Parquet
  exports; SQLite audit mirror; environment/model/data manifest; SHA-256 artifact
  integrity verification, FastAPI endpoints, local run queue, and Next.js dashboard.
- Full synthetic dataset and chronological offline runs with complete reasoning and reconciliation
  traces.

## External gates intentionally not fabricated

- Demo-video hosting remains an external submission step. The private data-preparation repository
  is intentionally not published; only its approved dataset payload is eligible for Hugging Face.
- Muse, Claude, and Gemini live smoke calls require `MODEL_API_KEY`, `ANTHROPIC_API_KEY`, and
  `GOOGLE_API_KEY`. Missing credentials fail before a run starts. OpenAI authentication was tested
  with a bounded two-month, 50-call replay and an exact no-key recorded replay.
- No rights-cleared real historical corpus was available. The shipped three-year profile is fully
  synthetic and labeled on every record and in the dashboard.

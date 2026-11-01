# Agent skills and tools

The runtime separates three responsibilities:

1. Deterministic inference computes reproducible market and company priors from every point-in-time
   data modality.
2. The LLM selects a versioned skill according to expected information gain.
3. Python validates the request and executes only the tools allowlisted by that skill.

Skills live in `src/portfolio_agent/agents/skill_library`. Each Markdown file contains versioned
metadata, selection guidance, an allowed-tool list, and a detailed investment-research playbook.
The registry validates the library at startup. Full instructions are loaded into model context only
when the skill is used.

| Skill | Scope | Tools |
| --- | --- | --- |
| `market-regime` | Global | Benchmark, factors, volatility, drawdown |
| `company-deep-dive` | Company | Price history, dated news, fundamentals |
| `peer-relative-value` | Company | Sector peers, fundamentals |
| `forecast-calibration` | Global | Fully matured forecast errors |
| `portfolio-awareness` | Portfolio | Holdings, baseline, sector exposure |
| `cross-sectional-reconciliation` | Cross-section | Mandatory reasoning playbook |
| `portfolio-policy` | Portfolio | Bounded experiment playbook |

The model cannot add tools, access future rows, execute arbitrary code, submit unrestricted weights,
or bypass reconciliation. Skill executions, tool observations, deterministic inferences, and model
traces are stored as immutable run artifacts and mirrored into SQLite.

Offline mock mode follows a deterministic skill sequence to test the runtime. Live mode lets the
selected model choose skills dynamically. Recorded mode replays previously captured structured
responses through the same validation boundary.

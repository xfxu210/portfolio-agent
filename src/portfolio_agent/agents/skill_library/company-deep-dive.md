---
name: company-deep-dive
version: "1.0"
description: Integrate company history, dated news and reported fundamentals into a causal return thesis.
when_to_use: Use for decision-relevant companies whose forecast is uncertain, extreme, stale, or contradicted by evidence.
target_scope: company
tools:
  - company_history
  - news_context
  - fundamentals_context
---
# Company deep-dive skill

Build a three-month thesis from independent evidence channels.

1. Separate durable fundamental changes from price momentum and one-off headlines.
2. Deduplicate repeated coverage of the same event; article count is not evidence strength.
3. Prefer newly reported fundamentals and dated catalysts that can matter inside 63 sessions.
4. Explicitly identify contradictions between price behavior, news, and fundamentals.
5. Revise the forecast only when the new evidence changes the probability-weighted return.
6. Preserve uncertainty and cite only supplied evidence identifiers.


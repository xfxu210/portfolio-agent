# 2–3 Minute Demo Script

1. Open the dashboard and identify `synthetic-full-v1`, its hash, retrospective provenance,
   model identity, and the 15/10 development-holdout split.
2. Show the three realized, after-cost curves. Point out that the adaptive strategy may lose;
   there is no hardcoded success banner.
3. Select February 2024. Show the evidence-bound experiment cards, ten-stock allocation,
   frozen projected fan, and matured prediction-versus-actual result.
4. Select May 2024. Show that completed forecast error caused confidence scaling to shrink,
   creating a new strategy version without changing the frozen meta-policy.
5. Select the final month. Explain why its 63-session actual result remains pending.
6. Open the exported report and run `portfolio-agent verify runs/demo`.
7. Show `portfolio-agent data validate` and the exact dataset hash.
8. If Muse credentials are available, show one genuine short live trace and then replay the
   captured output in recorded mode without a key.


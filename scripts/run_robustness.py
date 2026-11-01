from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_agent.agents.providers.mock import MockProvider
from portfolio_agent.config import AgentConfig, ReplayConfig
from portfolio_agent.data import DatasetStore
from portfolio_agent.evaluation.outer_replay import ReplayEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fees", type=float, nargs="+", default=[0.0, 10.0, 25.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    store = DatasetStore.load(args.dataset)
    results = []
    for seed in args.seeds:
        for fee_bps in args.fees:
            name = f"seed-{seed}-fee-{fee_bps:g}"
            config = ReplayConfig(
                seed=seed,
                fee_bps=fee_bps,
                agent=AgentConfig(mode="mock"),
            )
            summary = ReplayEngine(store, MockProvider(), config, args.out / name).run()
            results.append(
                {
                    "seed": seed,
                    "fee_bps": fee_bps,
                    "run_id": summary["run_id"],
                    "metrics": summary["metrics"],
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "robustness.json").write_text(
        json.dumps(
            {
                "dataset_hash": store.dataset_hash,
                "warning": "All positive, negative, and inconclusive results are retained.",
                "runs": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"completed {len(results)} robustness runs")


if __name__ == "__main__":
    main()

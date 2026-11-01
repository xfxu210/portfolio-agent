from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_agent.agents.providers.mock import MockProvider
from portfolio_agent.config import ReplayConfig
from portfolio_agent.data import DatasetStore
from portfolio_agent.evaluation.outer_replay import ReplayEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = ReplayEngine(
        DatasetStore.load(args.dataset), MockProvider(), ReplayConfig(), args.out
    ).run()
    print(summary)


if __name__ == "__main__":
    main()

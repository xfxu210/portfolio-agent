from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_agent.data import DatasetStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    store = DatasetStore.load(args.dataset)
    print(f"valid {store.manifest.dataset_version}: {store.manifest.coverage}")


if __name__ == "__main__":
    main()

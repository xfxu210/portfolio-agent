import os
from pathlib import Path

import pytest

from portfolio_agent.data import DatasetStore


def test_supplied_synthetic_full_contract_when_available() -> None:
    configured = os.environ.get("PORTFOLIO_AGENT_FULL_DATASET")
    if not configured:
        pytest.skip("PORTFOLIO_AGENT_FULL_DATASET not configured")
    store = DatasetStore.load(Path(configured))
    assert store.manifest.profile == "synthetic-full"
    assert int(store.manifest.coverage["sessions"]) == 756
    assert int(store.manifest.coverage["assets"]) == 40
    assert len(store.decision_dates()) >= 24

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from portfolio_agent.data import DatasetStore
from portfolio_agent.portfolio.signals import build_signal_table

DATASET = Path("examples/smoke")


def test_loader_enforces_point_in_time_research() -> None:
    store = DatasetStore.load(DATASET)
    early = store.research_as_of(date(2024, 1, 3))
    later = store.research_as_of(date(2025, 1, 3))
    assert len(later) >= len(early)
    assert all(row.available_at.date() <= date(2024, 1, 3) for row in early.itertuples())


def test_dataset_has_expected_full_coverage() -> None:
    store = DatasetStore.load(DATASET)
    assert store.manifest.coverage["sessions"] == 400
    assert store.manifest.coverage["assets"] == 12
    assert len(store.decision_dates()) >= 6


def test_future_article_mutation_cannot_change_earlier_signal() -> None:
    store = DatasetStore.load(DATASET)
    day = date(2025, 1, 2)
    before = build_signal_table(store, day, research_blend=0.5).reset_index(drop=True)
    future = store.articles.tail(1).copy()
    future["record_id"] = "future-injected"
    future["event_id"] = "future-injected"
    future["published_at"] = pd.Timestamp("2030-01-01T12:00:00Z")
    future["available_at"] = pd.Timestamp("2030-01-01T12:05:00Z")
    mutated = replace(store, articles=pd.concat([store.articles, future], ignore_index=True))
    after = build_signal_table(mutated, day, research_blend=0.5).reset_index(drop=True)
    assert_frame_equal(before, after)

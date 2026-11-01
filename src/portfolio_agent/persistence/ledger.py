from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class SQLiteLedger:
    """Portable queryable mirror of the immutable JSONL run artifacts."""

    TABLES = {
        "monthly_decisions": "monthly_decisions.jsonl",
        "experiments": "experiments.jsonl",
        "model_requests": "traces.jsonl",
        "company_analyses": "company_analysis.jsonl",
        "stock_forecasts": "stock_forecasts.jsonl",
        "projections": "projections.jsonl",
        "forecast_outcomes": "forecast_outcomes.jsonl",
    }

    def __init__(self, path: Path) -> None:
        self.path = path

    def capture(self, root: Path, summary: dict) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(run_id TEXT PRIMARY KEY, summary_json TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?)",
                (summary["run_id"], json.dumps(summary, sort_keys=True)),
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS state_transitions "
                "(sequence INTEGER PRIMARY KEY, state TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            connection.execute("DELETE FROM state_transitions")
            connection.execute(
                "INSERT INTO state_transitions VALUES (?, ?, ?)",
                (0, "completed", json.dumps({"run_id": summary["run_id"]}, sort_keys=True)),
            )
            for table, filename in self.TABLES.items():
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} "
                    "(sequence INTEGER PRIMARY KEY, payload_json TEXT NOT NULL)"
                )
                connection.execute(f"DELETE FROM {table}")
                path = root / filename
                if not path.exists():
                    continue
                rows = [
                    (index, json.dumps(json.loads(line), sort_keys=True))
                    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
                    if line
                ]
                connection.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)
            connection.commit()

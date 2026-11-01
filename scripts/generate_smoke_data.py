from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n",
        encoding="utf-8",
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def at_close(day: pd.Timestamp, delay: int = 0) -> datetime:
    return datetime.combine(day.date(), time(21), tzinfo=UTC) + timedelta(minutes=delay)


def generate(output: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2024-01-02", periods=400)
    symbols = [f"SMOKE{i:02d}" for i in range(1, 13)]
    returns = rng.normal(0.00025, 0.012, (len(sessions), len(symbols)))
    prices = rng.uniform(40, 120, len(symbols)) * np.exp(np.cumsum(np.log1p(returns), axis=0))
    universe = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "company_name": f"Smoke Company {index:02d}",
                "sector": ("Technology", "Healthcare", "Financials")[index % 3],
                "eligible_from": sessions[0].date(),
                "eligible_until": None,
                "universe_formation_date": sessions[0].date(),
                "source_id": "smoke-generator-v1",
                "available_at": at_close(sessions[0], -360),
                "currency": "USD",
            }
            for index, symbol in enumerate(symbols)
        ]
    )
    price_rows = [
        {
            "date": day.date(),
            "symbol": symbol,
            "adjusted_close": round(float(prices[day_index, symbol_index]), 8),
            "available_at": at_close(day, 15),
            "currency": "USD",
            "adjustment_method": "synthetic-total-return",
            "source_id": "smoke-generator-v1",
        }
        for day_index, day in enumerate(sessions)
        for symbol_index, symbol in enumerate(symbols)
    ]
    benchmark_returns = returns.mean(axis=1)
    benchmark = 100 * np.exp(np.cumsum(np.log1p(benchmark_returns)))
    benchmark_rows = [
        {
            "date": day.date(),
            "symbol": "SMOKE-BENCH",
            "adjusted_close": round(float(benchmark[index]), 8),
            "available_at": at_close(day, 15),
            "currency": "USD",
            "adjustment_method": "synthetic-total-return",
            "source_id": "smoke-generator-v1",
        }
        for index, day in enumerate(sessions)
    ]
    factor_rows = [
        {
            "date": day.date(),
            "factor_name": name,
            "value": float(value),
            "available_at": at_close(day, 20),
            "source_id": "smoke-generator-v1",
            "unit": "decimal_daily_return",
            "frequency": "daily",
        }
        for index, day in enumerate(sessions)
        for name, value in (
            ("market_return", benchmark_returns[index]),
            ("risk_free_return", 0.04 / 252),
        )
    ]
    article_rows = []
    for index, day_index in enumerate(range(20, len(sessions), 18)):
        symbol = symbols[index % len(symbols)]
        positive = index % 2 == 0
        article_rows.append(
            {
                "record_id": f"smoke-article-{index:04d}",
                "symbols": [symbol],
                "title": f"{symbol} synthetic {'positive' if positive else 'negative'} event",
                "summary": "Synthetic smoke research with no future return labels.",
                "source_id": "smoke-generator-v1",
                "source_url": None,
                "event_type": "corporate_update",
                "published_at": at_close(sessions[day_index], -60),
                "available_at": at_close(sessions[day_index], -55),
                "license": "synthetic-generated",
                "is_synthetic": True,
                "provenance": "smoke-generator-v1",
                "event_id": f"smoke-event-{index:04d}",
                "exploratory_only": False,
            }
        )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "metric": "operating_margin",
                "value": round(0.08 + symbol_index * 0.01 + quarter * 0.001, 6),
                "unit": "decimal",
                "period_end": sessions[day_index - 10].date(),
                "filing_published_at": at_close(sessions[day_index], -120),
                "available_at": at_close(sessions[day_index], -115),
                "source_id": "smoke-generator-v1",
                "data_revision": f"original-{quarter}",
                "retrospective_only": False,
            }
            for symbol_index, symbol in enumerate(symbols)
            for quarter, day_index in enumerate(range(62, len(sessions), 63), start=1)
        ]
    )
    frames = {
        "universe/universe.parquet": universe,
        "market/prices.parquet": pd.DataFrame(price_rows),
        "market/benchmark.parquet": pd.DataFrame(benchmark_rows),
        "market/factors.parquet": pd.DataFrame(factor_rows),
        "research/fundamentals.parquet": fundamentals,
        "research/articles/year=2024/month=01/part-000.parquet": pd.DataFrame(article_rows),
    }
    for relative, frame in frames.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
    write_json(
        output / "portfolio/initial_portfolio.json",
        {
            "schema_version": "1.0.0",
            "portfolio_id": "smoke-reference",
            "as_of": sessions[0].date().isoformat(),
            "base_currency": "USD",
            "benchmark": "SMOKE-BENCH",
            "initial_nav": 100000.0,
            "holdings": [{"symbol": symbol, "weight": 0.1} for symbol in symbols[:10]],
            "constraints": {
                "holding_count": 10,
                "min_weight": 0.02,
                "max_weight": 0.25,
                "max_one_way_turnover": 0.20,
                "long_only": True,
                "allow_leverage": False,
            },
            "is_synthetic": True,
            "data_version": "smoke-v1",
        },
    )
    generated_at = "2026-10-31T00:00:00+00:00"
    write_json(
        output / "reports/rights_summary.json",
        {
            "schema_version": "1.0.0",
            "generated_at": generated_at,
            "deny_unknown": True,
            "decisions": [
                {
                    "source_id": "smoke-generator-v1",
                    "field": "*",
                    "decision": "publish",
                    "rationale": "Entirely generated synthetic fixture.",
                    "license_reference": "synthetic-generated",
                }
            ],
        },
    )
    files = {
        str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(
        output / "manifest.json",
        {
            "schema_version": "1.0.0",
            "dataset_version": "smoke-v1",
            "profile": "smoke",
            "generated_at": generated_at,
            "is_synthetic": True,
            "files": files,
            "coverage": {
                "sessions": len(sessions),
                "assets": len(symbols),
                "articles": len(article_rows),
                "fundamentals": len(fundamentals),
                "start_date": sessions[0].date().isoformat(),
                "end_date": sessions[-1].date().isoformat(),
            },
            "rights_summary_path": "reports/rights_summary.json",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("output must be absent or empty")
    generate(args.out, args.seed)
    print(f"generated smoke fixture at {args.out}")


if __name__ == "__main__":
    main()

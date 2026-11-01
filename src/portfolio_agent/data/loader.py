from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from portfolio_agent.data.hf_download import SOURCE_METADATA_FILE
from portfolio_agent.domain.schemas import DatasetManifest, InitialPortfolio


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    repo_id: str = Field(min_length=3)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DatasetStore:
    root: Path
    manifest: DatasetManifest
    universe: pd.DataFrame
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    factors: pd.DataFrame
    articles: pd.DataFrame
    fundamentals: pd.DataFrame
    initial_portfolio: InitialPortfolio
    source: DatasetSource | None = None

    @classmethod
    def load(cls, root: Path) -> DatasetStore:
        manifest = DatasetManifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest.files.items():
            path = root / relative
            if not path.is_file():
                raise ValueError(f"manifest file missing: {relative}")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"checksum mismatch: {relative}")
        article_paths = sorted((root / "research/articles").glob("year=*/month=*/*.parquet"))
        if not article_paths:
            raise ValueError("dataset contains no article partitions")
        initial = InitialPortfolio.model_validate_json(
            (root / "portfolio/initial_portfolio.json").read_text(encoding="utf-8")
        )
        source_path = root / SOURCE_METADATA_FILE
        source = (
            DatasetSource.model_validate_json(source_path.read_text(encoding="utf-8"))
            if source_path.is_file()
            else None
        )
        store = cls(
            root=root,
            manifest=manifest,
            universe=pd.read_parquet(root / "universe/universe.parquet"),
            prices=pd.read_parquet(root / "market/prices.parquet"),
            benchmark=pd.read_parquet(root / "market/benchmark.parquet"),
            factors=pd.read_parquet(root / "market/factors.parquet"),
            articles=pd.concat(
                [pd.read_parquet(path) for path in article_paths], ignore_index=True
            ),
            fundamentals=pd.read_parquet(root / "research/fundamentals.parquet"),
            initial_portfolio=initial,
            source=source,
        )
        store._validate_integrity()
        return store

    def _validate_integrity(self) -> None:
        if self.prices.duplicated(["date", "symbol"]).any():
            raise ValueError("duplicate price key")
        if self.articles["record_id"].duplicated().any():
            raise ValueError("duplicate article record_id")
        universe_symbols = set(self.universe["symbol"])
        unknown = set(self.prices["symbol"]) - universe_symbols
        if unknown:
            raise ValueError(f"prices reference unknown symbols: {sorted(unknown)}")

    @property
    def dataset_hash(self) -> str:
        encoded = json.dumps(self.manifest.files, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def cutoff(day: date) -> datetime:
        return datetime.combine(day, time(23, 59, 59), tzinfo=UTC)

    def eligible_symbols(self, day: date) -> list[str]:
        until = self.universe["eligible_until"]
        mask = (self.universe["eligible_from"] <= day) & (until.isna() | (day < until))
        return sorted(self.universe.loc[mask, "symbol"].tolist())

    def price_history(self, day: date, symbols: list[str] | None = None) -> pd.DataFrame:
        cutoff = self.cutoff(day)
        mask = (self.prices["date"] <= day) & (self.prices["available_at"] <= cutoff)
        if symbols is not None:
            mask &= self.prices["symbol"].isin(symbols)
        return self.prices.loc[mask].sort_values(["date", "symbol"]).copy()

    def research_as_of(self, day: date, symbols: list[str] | None = None) -> pd.DataFrame:
        cutoff = self.cutoff(day)
        mask = (
            (self.articles["published_at"] <= cutoff)
            & (self.articles["available_at"] <= cutoff)
            & ~self.articles["exploratory_only"]
        )
        frame = self.articles.loc[mask].copy()
        if symbols is not None:
            wanted = set(symbols)
            frame = frame[frame["symbols"].map(lambda values: bool(wanted.intersection(values)))]
        return frame.sort_values(["published_at", "record_id"])

    def fundamentals_as_of(self, day: date, symbols: list[str] | None = None) -> pd.DataFrame:
        cutoff = self.cutoff(day)
        mask = (self.fundamentals["available_at"] <= cutoff) & ~self.fundamentals[
            "retrospective_only"
        ]
        if symbols is not None:
            mask &= self.fundamentals["symbol"].isin(symbols)
        return self.fundamentals.loc[mask].sort_values(["available_at", "symbol"]).copy()

    def decision_dates(self, warmup_sessions: int = 252) -> list[date]:
        sessions = sorted(self.prices["date"].unique())
        eligible = sessions[warmup_sessions:]
        frame = pd.DataFrame({"date": eligible})
        frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M")
        return frame.groupby("month", sort=True)["date"].min().tolist()

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from typing import Any

from portfolio_agent.agents.providers.base import LLMProvider, ProviderError
from portfolio_agent.data.loader import DatasetStore
from portfolio_agent.domain.schemas import CompanyAnalysis, StockForecast


class ResearchService:
    def __init__(
        self,
        store: DatasetStore,
        provider: LLMProvider,
        trace: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.trace = trace
        self.cache: dict[tuple[str, str, str, date, str, str], CompanyAnalysis | StockForecast] = {}

    def _record(self, stage: str, request_id: str, result: Any, day: date) -> None:
        if self.trace is None:
            return
        self.trace(
            {
                "stage": stage,
                "request_id": request_id,
                "provider": result.provider,
                "model": result.model,
                "provider_request_id": result.request_id,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "output_hash": result.raw_hash,
                "normalized": result.value.model_dump(mode="json"),
                "decision_as_of": day.isoformat(),
            }
        )

    def analyze_and_forecast(
        self, day: date, symbol: str, quant_prior: float
    ) -> tuple[CompanyAnalysis, StockForecast]:
        evidence = self.store.research_as_of(day, [symbol]).tail(8)
        fundamentals = self.store.fundamentals_as_of(day, [symbol]).tail(8)
        article_ids = tuple(evidence["record_id"].tolist())
        fundamental_ids = tuple(
            f"fundamental:{row.symbol}:{row.metric}:{row.available_at.isoformat()}:{row.data_revision}"
            for row in fundamentals.itertuples()
        )
        evidence_ids = (*article_ids, *fundamental_ids)
        cutoff = datetime.combine(day, time(23, 59, 59), tzinfo=UTC)
        context = {
            "decision_as_of": cutoff.isoformat(),
            "symbol": symbol,
            "evidence_ids": evidence_ids,
            "quant_prior": float(quant_prior),
            "evidence": [
                {
                    "id": row.record_id,
                    "title": row.title,
                    "summary": row.summary,
                    "published_at": row.published_at.isoformat(),
                }
                for row in evidence.itertuples()
            ],
            "fundamentals": [
                {
                    "id": evidence_id,
                    "metric": row.metric,
                    "value": row.value,
                    "unit": row.unit,
                    "period_end": row.period_end.isoformat(),
                    "available_at": row.available_at.isoformat(),
                }
                for evidence_id, row in zip(
                    fundamental_ids, fundamentals.itertuples(), strict=True
                )
            ],
            "warning": "Retrospective historical simulation; use only supplied evidence.",
        }
        context_hash = json.dumps(context, sort_keys=True, default=str)
        analysis_key = (
            "company_analysis",
            self.provider.provider,
            self.provider.model,
            day,
            symbol,
            context_hash,
        )
        forecast_key = (
            "stock_forecast",
            self.provider.provider,
            self.provider.model,
            day,
            symbol,
            context_hash,
        )
        if analysis_key not in self.cache:
            request_id = f"analysis:{day}:{symbol}"
            result = self.provider.generate_structured(
                task="company_analysis",
                messages=[{"role": "user", "content": json.dumps(context, default=str)}],
                response_model=CompanyAnalysis,
                request_id=request_id,
            )
            unknown = set(result.value.evidence_ids) - set(evidence_ids)
            if unknown:
                raise ProviderError(
                    "ungrounded_evidence",
                    f"company analysis cited unknown evidence IDs: {sorted(unknown)}",
                )
            if result.value.symbol != symbol or result.value.decision_as_of != cutoff:
                raise ProviderError(
                    "provenance_mismatch",
                    "company analysis symbol/decision_as_of does not match request context",
                )
            self._record("company_analysis", request_id, result, day)
            self.cache[analysis_key] = result.value
        analysis = self.cache[analysis_key]
        assert isinstance(analysis, CompanyAnalysis)
        if forecast_key not in self.cache:
            forecast_context = {**context, "analysis_id": analysis.analysis_id}
            request_id = f"forecast:{day}:{symbol}"
            result = self.provider.generate_structured(
                task="stock_forecast",
                messages=[{"role": "user", "content": json.dumps(forecast_context, default=str)}],
                response_model=StockForecast,
                request_id=request_id,
            )
            unknown = set(result.value.evidence_ids) - set(evidence_ids)
            if unknown:
                raise ProviderError(
                    "ungrounded_evidence",
                    f"stock forecast cited unknown evidence IDs: {sorted(unknown)}",
                )
            if (
                result.value.symbol != symbol
                or result.value.decision_as_of != cutoff
                or result.value.analysis_id != analysis.analysis_id
            ):
                raise ProviderError(
                    "provenance_mismatch",
                    "stock forecast symbol/date/analysis linkage does not match request context",
                )
            self._record("stock_forecast", request_id, result, day)
            self.cache[forecast_key] = result.value
        forecast = self.cache[forecast_key]
        assert isinstance(forecast, StockForecast)
        return analysis, forecast

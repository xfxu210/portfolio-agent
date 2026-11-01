from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from portfolio_agent.domain.schemas import (
    CompanyAnalysis,
    ExperimentProposal,
    ExperimentType,
    StockForecast,
)

from .base import StructuredResult

T = TypeVar("T", bound=BaseModel)


class MockProvider:
    provider = "mock"
    model = "deterministic-v1"

    def generate_structured(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        request_id: str,
    ) -> StructuredResult[T]:
        digest = hashlib.sha256((task + str(messages)).encode()).hexdigest()
        try:
            context = json.loads(messages[-1]["content"]) if messages else {}
        except json.JSONDecodeError:
            context = {}
        now = datetime.fromisoformat(
            str(context.get("decision_as_of", "2026-10-31T00:00:00+00:00")).replace("Z", "+00:00")
        ).astimezone(UTC)
        symbol = context.get("symbol", "SYNTHETIC")
        evidence_ids = tuple(context.get("evidence_ids", ()))
        if response_model is CompanyAnalysis:
            payload = {
                "analysis_id": context.get("analysis_id", f"analysis-{digest[:12]}"),
                "decision_as_of": now,
                "created_at": now,
                "symbol": symbol,
                "provider": "mock",
                "model": self.model,
                "model_settings_hash": digest,
                "prompt_version": "v1",
                "evidence_ids": evidence_ids,
                "bull_drivers": ("deterministic positive scenario",),
                "bear_drivers": ("deterministic negative scenario",),
                "key_catalysts": (),
                "risks": ("synthetic mock output",),
                "confidence": 0.5,
                "rationale": "Deterministic mock analysis.",
                "retrospective_model": True,
            }
        elif response_model is StockForecast:
            payload = {
                "forecast_id": f"forecast-{digest[:12]}",
                "decision_as_of": now,
                "created_at": now,
                "symbol": symbol,
                "expected_return_3m": float(context.get("quant_prior", 0.02)),
                "confidence": 0.5,
                "rationale": "Deterministic mock forecast.",
                "analysis_id": context.get("analysis_id", f"analysis-{digest[:12]}"),
                "evidence_ids": evidence_ids,
                "provider": "mock",
                "model": self.model,
                "model_settings_hash": digest,
                "prompt_version": "v1",
                "is_retrospective": True,
            }
        elif response_model is ExperimentProposal:
            iteration = int(context.get("iteration", 0))
            matured_feedback = context.get("matured_feedback", [])
            calibrated = len(matured_feedback) >= 5
            recent_error = (
                float(
                    sum(abs(item["forecast_error"]) for item in matured_feedback[-10:])
                    / len(matured_feedback[-10:])
                )
                if calibrated
                else 0.0
            )
            feedback_risk_aversion = 4.0 if recent_error > 0.05 else 2.0
            feedback_confidence_scale = 0.5 if recent_error > 0.05 else 0.75
            choices = (
                (
                    ExperimentType.RESEARCH_BLEND,
                    {"research_blend": (0.25, 0.5, 0.75)[iteration % 3]},
                    "Test whether research coverage improves ranking.",
                ),
                (
                    ExperimentType.RISK_AVERSION,
                    {"risk_aversion": feedback_risk_aversion},
                    "Test whether recent forecast error warrants more risk control.",
                ),
                (
                    ExperimentType.FORECAST_CONFIDENCE_SCALE,
                    {"confidence_scale": feedback_confidence_scale},
                    "Test shrinkage of retrospective model forecasts toward priors.",
                ),
            )
            kind, overrides, hypothesis = choices[iteration % len(choices)]
            payload = {
                "experiment_type": kind,
                "hypothesis": hypothesis,
                "evidence_ids": evidence_ids,
                "overrides": overrides,
                "expected_tradeoffs": ("May underperform in a different market regime.",),
            }
        else:
            raise TypeError(f"mock provider has no fixture for {response_model.__name__}")
        value = response_model.model_validate(payload)
        return StructuredResult(
            value=value,
            provider=self.provider,
            model=self.model,
            request_id=request_id,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            raw_hash=digest,
        )

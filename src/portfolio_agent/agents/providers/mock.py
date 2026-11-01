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
    ForecastAction,
    ForecastAgentDecision,
    ForecastReconciliation,
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
        elif response_model is ForecastAgentDecision:
            step = int(context.get("step", 0))
            shortlist = tuple(context.get("shortlist", ()))
            choices = (
                (
                    ForecastAction.INSPECT_MARKET_REGIME,
                    (),
                    "What market regime and common-factor backdrop constrain all forecasts?",
                    0.8,
                ),
                (
                    ForecastAction.INSPECT_PEER_CONTEXT,
                    shortlist[:5],
                    "Are the highest-priority companies consistent with their sector peers?",
                    0.65,
                ),
                (
                    ForecastAction.INSPECT_CALIBRATION,
                    (),
                    "What bias and error are visible in fully matured prior forecasts?",
                    0.55,
                ),
                (
                    ForecastAction.INSPECT_COMPANY_HISTORY,
                    shortlist[:3],
                    "Do recent multi-horizon returns contradict the current forecasts?",
                    0.45,
                ),
                (
                    ForecastAction.REVISE_FORECASTS,
                    shortlist[:2],
                    "Revisit the most decision-relevant forecasts using the new context.",
                    0.35,
                ),
                (
                    ForecastAction.RECONCILE_FORECASTS,
                    (),
                    "Reconcile every company forecast into one coherent cross-section.",
                    0.25,
                ),
                (
                    ForecastAction.FINISH,
                    (),
                    "Is the reconciled forecast set complete enough to use?",
                    0.0,
                ),
            )
            action, symbols, question, information_gain = choices[min(step, len(choices) - 1)]
            payload = {
                "action": action,
                "symbols": symbols,
                "question": question,
                "rationale": "Deterministic mock policy exercises the bounded agent loop.",
                "expected_information_gain": information_gain,
            }
        elif response_model is ForecastReconciliation:
            raw_forecasts = list(context.get("company_forecasts", ()))
            priors = context.get("quant_priors", {})
            observations = context.get("tool_observations", [])
            market_observation = next(
                (
                    item.get("result", {})
                    for item in observations
                    if item.get("tool") == ForecastAction.INSPECT_MARKET_REGIME
                ),
                {},
            )
            calibration = next(
                (
                    item.get("result", {})
                    for item in observations
                    if item.get("tool") == ForecastAction.INSPECT_CALIBRATION
                ),
                {},
            )
            market_expected = float(market_observation.get("benchmark_return_63d") or 0.0)
            market_expected = max(-1.0, min(3.0, market_expected))
            error = float(calibration.get("mean_absolute_error") or 0.0)
            calibration_intercept = float(
                calibration.get("standalone_calibration_intercept") or 0.0
            )
            calibration_slope = float(
                calibration.get("standalone_calibration_slope") or 1.0
            )
            calibration_reliability = float(
                calibration.get("calibration_reliability") or 0.0
            )
            reconciled = []
            for item in raw_forecasts:
                symbol = str(item["symbol"])
                raw = float(item["expected_return_3m"])
                prior = float(priors.get(symbol, raw))
                standalone = 0.75 * raw + 0.25 * prior
                calibrated = calibration_intercept + calibration_slope * standalone
                adjusted = (
                    (1 - calibration_reliability) * standalone
                    + calibration_reliability * calibrated
                )
                adjusted = max(-1.0, min(3.0, adjusted))
                confidence = max(0.05, min(1.0, float(item["confidence"]) * (1 - error)))
                reconciled.append(
                    {
                        "symbol": symbol,
                        "source_forecast_id": item["forecast_id"],
                        "raw_expected_return_3m": raw,
                        "expected_return_3m": adjusted,
                        "confidence": confidence,
                        "bear_case_return": max(-1.0, adjusted - 0.08),
                        "bull_case_return": min(3.0, adjusted + 0.08),
                        "adjustment_rationale": (
                            "Shrink the standalone estimate toward its prior, then apply the "
                            "regularized intercept and slope learned only from matured forecasts."
                        ),
                        "consistency_flags": (),
                    }
                )
            reconciled.sort(key=lambda item: (-item["expected_return_3m"], item["symbol"]))
            for rank, item in enumerate(reconciled, start=1):
                item["relative_rank"] = rank
            payload = {
                "reconciliation_id": "pending",
                "decision_as_of": now,
                "provider": self.provider,
                "model": self.model,
                "market_regime": str(market_observation.get("regime", "unclassified")),
                "market_expected_return_3m": market_expected,
                "common_factor_assumptions": (
                    "All company estimates share the observed market-regime component.",
                ),
                "forecasts": tuple(reconciled),
                "cross_sectional_risks": (
                    "Common synthetic factor exposure can make standalone forecasts correlated.",
                ),
                "confidence": max(0.05, min(1.0, 0.6 * (1 - error))),
                "rationale": (
                    "Jointly shrink, calibrate, and rank every supplied company forecast."
                ),
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

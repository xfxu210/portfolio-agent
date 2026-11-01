from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from portfolio_agent.agents.providers.base import LLMProvider, ProviderError
from portfolio_agent.agents.research import ResearchService
from portfolio_agent.config import ReasoningConfig
from portfolio_agent.data.loader import DatasetStore
from portfolio_agent.domain.schemas import (
    CompanyAnalysis,
    ForecastAction,
    ForecastAgentDecision,
    ForecastReconciliation,
    StockForecast,
)


@dataclass(frozen=True)
class ForecastingResult:
    analyses: dict[str, CompanyAnalysis]
    forecasts: dict[str, StockForecast]
    reconciliation: ForecastReconciliation
    actions: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]


class ForecastingTools:
    """Point-in-time deterministic tools available to the forecasting agent."""

    def __init__(self, store: DatasetStore, signals: pd.DataFrame) -> None:
        self.store = store
        self.signals = signals.set_index("symbol")

    @staticmethod
    def _return(series: pd.Series, sessions: int) -> float | None:
        values = series.dropna().tail(sessions + 1)
        if len(values) < sessions + 1:
            return None
        return float(values.iloc[-1] / values.iloc[0] - 1)

    @staticmethod
    def _drawdown(series: pd.Series) -> float | None:
        values = series.dropna()
        if values.empty:
            return None
        return float((values / values.cummax() - 1).min())

    def market_regime(self, day: date) -> dict[str, Any]:
        cutoff = self.store.cutoff(day)
        frame = self.store.benchmark[
            (self.store.benchmark["date"] <= day)
            & (self.store.benchmark["available_at"] <= cutoff)
        ].sort_values("date")
        prices = frame["adjusted_close"].astype(float)
        returns_63 = self._return(prices, 63)
        returns_126 = self._return(prices, 126)
        daily = prices.pct_change().dropna().tail(63)
        volatility = float(daily.std(ddof=0) * np.sqrt(252)) if len(daily) else None
        drawdown = self._drawdown(prices.tail(127))
        if returns_63 is None:
            regime = "insufficient_history"
        elif returns_63 < -0.08:
            regime = "drawdown"
        elif volatility is not None and volatility > 0.25:
            regime = "high_volatility"
        elif returns_63 > 0.08:
            regime = "risk_on"
        else:
            regime = "range_bound"
        factors = self.store.factors[
            (self.store.factors["date"] <= day)
            & (self.store.factors["available_at"] <= cutoff)
        ].sort_values("date")
        factor_snapshot = {
            str(name): float(group.tail(21)["value"].astype(float).mean())
            for name, group in factors.groupby("factor_name")
        }
        return {
            "regime": regime,
            "benchmark_return_21d": self._return(prices, 21),
            "benchmark_return_63d": returns_63,
            "benchmark_return_126d": returns_126,
            "benchmark_volatility_63d": volatility,
            "benchmark_drawdown_126d": drawdown,
            "factor_21d_means": factor_snapshot,
        }

    def company_history(self, day: date, symbols: tuple[str, ...]) -> dict[str, Any]:
        history = self.store.price_history(day, list(symbols))
        pivot = history.pivot(index="date", columns="symbol", values="adjusted_close")
        result: dict[str, Any] = {}
        for symbol in symbols:
            series = pivot[symbol].astype(float) if symbol in pivot else pd.Series(dtype=float)
            daily = series.pct_change().dropna().tail(63)
            result[symbol] = {
                "return_21d": self._return(series, 21),
                "return_63d": self._return(series, 63),
                "return_126d": self._return(series, 126),
                "volatility_63d": (
                    float(daily.std(ddof=0) * np.sqrt(252)) if len(daily) else None
                ),
                "drawdown_126d": self._drawdown(series.tail(127)),
            }
        return result

    def peer_context(self, symbols: tuple[str, ...]) -> dict[str, Any]:
        sectors = self.store.universe.set_index("symbol")["sector"].astype(str)
        result: dict[str, Any] = {}
        for symbol in symbols:
            if symbol not in self.signals.index:
                continue
            sector = sectors.get(symbol, "unknown")
            peer_symbols = [
                peer
                for peer in self.signals.index
                if sectors.get(peer, "unknown") == sector
            ]
            peers = self.signals.loc[peer_symbols]
            row = self.signals.loc[symbol]
            result[symbol] = {
                "sector": sector,
                "rank_score": float(row["rank_score"]),
                "quant_score": float(row["quant_score"]),
                "research_score": float(row["research_score"]),
                "sector_peer_count": len(peer_symbols),
                "sector_median_rank_score": float(peers["rank_score"].median()),
                "sector_median_momentum": float(peers["momentum"].median()),
                "sector_rank": int(peers["rank_score"].rank(ascending=False)[symbol]),
            }
        return result

    @staticmethod
    def calibration(feedback: list[dict[str, Any]]) -> dict[str, Any]:
        if not feedback:
            return {
                "matured_forecasts": 0,
                "status": "insufficient_history",
            }
        feedback = sorted(
            feedback,
            key=lambda item: (str(item.get("matured_at", "")), str(item["symbol"])),
        )
        errors = np.array([float(item["forecast_error"]) for item in feedback], dtype=float)
        standalone = np.array(
            [
                float(item.get("raw_expected_return_3m", item["expected_return_3m"]))
                for item in feedback
            ],
            dtype=float,
        )
        realized = np.array(
            [float(item["realized_63_session_return"]) for item in feedback], dtype=float
        )
        def fit(values: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
            centered = values - float(np.mean(values))
            fitted_slope = float(
                np.mean(centered * (targets - float(np.mean(targets))))
                / (np.mean(centered**2) + 0.001)
            )
            fitted_slope = float(np.clip(fitted_slope, -1.0, 2.0))
            fitted_intercept = float(np.mean(targets) - fitted_slope * np.mean(values))
            return float(np.clip(fitted_intercept, -0.2, 0.2)), fitted_slope

        calibration_accepted = False
        validation_raw_mae: float | None = None
        validation_calibrated_mae: float | None = None
        if len(feedback) >= 40:
            split = max(20, int(len(feedback) * 0.7))
            trial_intercept, trial_slope = fit(standalone[:split], realized[:split])
            validation_raw_mae = float(
                np.mean(np.abs(realized[split:] - standalone[split:]))
            )
            validation_calibrated_mae = float(
                np.mean(
                    np.abs(
                        realized[split:]
                        - (trial_intercept + trial_slope * standalone[split:])
                    )
                )
            )
            calibration_accepted = validation_calibrated_mae < validation_raw_mae
        if calibration_accepted:
            intercept, slope = fit(standalone, realized)
            reliability = min(1.0, len(feedback) / 200)
        else:
            slope = 1.0
            intercept = 0.0
            reliability = 0.0
        directional = [
            np.sign(float(item["expected_return_3m"]))
            == np.sign(float(item["realized_63_session_return"]))
            for item in feedback
        ]
        by_symbol: dict[str, list[float]] = {}
        for item in feedback:
            by_symbol.setdefault(str(item["symbol"]), []).append(float(item["forecast_error"]))
        return {
            "matured_forecasts": len(feedback),
            "mean_absolute_error": float(np.mean(np.abs(errors))),
            "mean_bias_realized_minus_forecast": float(np.mean(errors)),
            "directional_accuracy": float(np.mean(directional)),
            "standalone_calibration_intercept": intercept,
            "standalone_calibration_slope": slope,
            "calibration_reliability": reliability,
            "calibration_accepted": calibration_accepted,
            "validation_raw_mae": validation_raw_mae,
            "validation_calibrated_mae": validation_calibrated_mae,
            "recent_errors": feedback[-20:],
            "symbol_bias": {
                symbol: float(np.mean(values)) for symbol, values in sorted(by_symbol.items())
            },
        }


class ForecastingAgent:
    """Bounded decide-act-observe agent that owns monthly return forecasting."""

    ACTIONS = {
        ForecastAction.INSPECT_MARKET_REGIME: (
            "Inspect benchmark, factors, volatility, and drawdown available at the cutoff."
        ),
        ForecastAction.INSPECT_PEER_CONTEXT: (
            "Compare selected companies with point-in-time sector peers."
        ),
        ForecastAction.INSPECT_COMPANY_HISTORY: (
            "Inspect multi-horizon returns, volatility, and drawdown for target companies."
        ),
        ForecastAction.INSPECT_CALIBRATION: (
            "Inspect only matured prior company-forecast errors."
        ),
        ForecastAction.REVISE_FORECASTS: (
            "Re-analyze up to five uncertain companies using accumulated observations."
        ),
        ForecastAction.RECONCILE_FORECASTS: (
            "Jointly reconcile every company forecast and produce a relative ranking."
        ),
        ForecastAction.FINISH: "Stop only after a complete reconciliation exists.",
    }

    def __init__(
        self,
        store: DatasetStore,
        provider: LLMProvider,
        research: ResearchService,
        reasoning: ReasoningConfig,
        append: Callable[[str, Any], str],
    ) -> None:
        self.store = store
        self.provider = provider
        self.research = research
        self.reasoning = reasoning
        self.append = append

    def _record_result(self, task: str, request_id: str, result: Any, day: date) -> None:
        self.append(
            "traces.jsonl",
            {
                "stage": task,
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
            },
        )

    @staticmethod
    def _forecast_state(
        analyses: dict[str, CompanyAnalysis],
        forecasts: dict[str, StockForecast],
        *,
        detailed: bool = False,
    ) -> list[dict[str, Any]]:
        values = []
        for symbol, forecast in sorted(forecasts.items()):
            value: dict[str, Any] = {
                "symbol": symbol,
                "analysis_id": analyses[symbol].analysis_id,
                "forecast_id": forecast.forecast_id,
                "expected_return_3m": forecast.expected_return_3m,
                "confidence": forecast.confidence,
                "primary_risks": analyses[symbol].risks[:2],
                "rationale": forecast.rationale[:300],
            }
            if detailed:
                value.update(
                    bull_drivers=analyses[symbol].bull_drivers,
                    bear_drivers=analyses[symbol].bear_drivers,
                    catalysts=analyses[symbol].key_catalysts,
                    evidence_ids=forecast.evidence_ids,
                    quality_flags=analyses[symbol].quality_flags,
                )
            values.append(value)
        return values

    def _decide(
        self,
        day: date,
        step: int,
        shortlist: list[str],
        analyses: dict[str, CompanyAnalysis],
        forecasts: dict[str, StockForecast],
        observations: list[dict[str, Any]],
        reconciliation: ForecastReconciliation | None,
    ) -> ForecastAgentDecision:
        context = {
            "decision_as_of": self.store.cutoff(day).isoformat(),
            "objective": (
                "Improve calibrated three-month return forecasts, resolve material uncertainty, "
                "and produce one coherent cross-company forecast set without future information."
            ),
            "step": step,
            "max_steps": self.reasoning.max_steps,
            "shortlist": shortlist,
            "available_actions": {key.value: value for key, value in self.ACTIONS.items()},
            "current_forecasts": self._forecast_state(analyses, forecasts),
            "observations": observations,
            "has_reconciliation": reconciliation is not None,
            "reconciliation": (
                reconciliation.model_dump(mode="json") if reconciliation is not None else None
            ),
            "instruction": (
                "Choose the single action with the highest expected information gain. Inspect or "
                "revise only when it could materially change a forecast. Reconcile all symbols "
                "before finishing."
            ),
        }
        request_id = f"forecast-agent:{day}:{step}"
        result = self.provider.generate_structured(
            task="forecast_agent_decision",
            messages=[{"role": "user", "content": json.dumps(context, default=str)}],
            response_model=ForecastAgentDecision,
            request_id=request_id,
        )
        self._record_result("forecast_agent_decision", request_id, result, day)
        value = result.value
        valid_symbols = tuple(symbol for symbol in value.symbols if symbol in set(shortlist))
        limit = (
            self.reasoning.max_revision_symbols
            if value.action == ForecastAction.REVISE_FORECASTS
            else 5
        )
        valid_symbols = valid_symbols[:limit]
        guarded = valid_symbols != value.symbols
        if guarded:
            if not valid_symbols and value.action in {
                ForecastAction.INSPECT_PEER_CONTEXT,
                ForecastAction.INSPECT_COMPANY_HISTORY,
                ForecastAction.REVISE_FORECASTS,
            }:
                valid_symbols = (shortlist[0],)
            value = value.model_copy(
                update={
                    "symbols": valid_symbols,
                    "rationale": (
                        f"{value.rationale} Runtime guard retained only eligible targets within "
                        "the configured per-action limit."
                    ),
                }
            )
        return value

    def _reconcile(
        self,
        day: date,
        analyses: dict[str, CompanyAnalysis],
        forecasts: dict[str, StockForecast],
        priors: dict[str, float],
        observations: list[dict[str, Any]],
        version: int,
    ) -> ForecastReconciliation:
        context = {
            "decision_as_of": self.store.cutoff(day).isoformat(),
            "objective": (
                "Reconcile all company forecasts jointly. Correct common-factor double counting, "
                "relative-value inconsistencies, confidence miscalibration, and implausible "
                "cross-sectional dispersion. Return every supplied symbol exactly once."
            ),
            "quant_priors": priors,
            "company_forecasts": self._forecast_state(
                analyses, forecasts, detailed=True
            ),
            "tool_observations": observations,
            "required_symbols": sorted(forecasts),
        }
        request_id = f"forecast-reconciliation:{day}:{version}"
        result = self.provider.generate_structured(
            task="forecast_reconciliation",
            messages=[{"role": "user", "content": json.dumps(context, default=str)}],
            response_model=ForecastReconciliation,
            request_id=request_id,
        )
        self._record_result("forecast_reconciliation", request_id, result, day)
        value = result.value
        expected_symbols = set(forecasts)
        actual_symbols = {item.symbol for item in value.forecasts}
        if actual_symbols != expected_symbols:
            raise ProviderError(
                "incomplete_reconciliation",
                f"reconciliation symbols differ: expected {sorted(expected_symbols)}, "
                f"received {sorted(actual_symbols)}",
            )
        reconciled_items = tuple(
            item.model_copy(
                update={
                    "source_forecast_id": forecasts[item.symbol].forecast_id,
                    "raw_expected_return_3m": forecasts[item.symbol].expected_return_3m,
                }
            )
            for item in value.forecasts
        )
        value = value.model_copy(update={"forecasts": reconciled_items})
        if value.decision_as_of != self.store.cutoff(day):
            raise ProviderError(
                "provenance_mismatch", "reconciliation decision date does not match the cutoff"
            )
        payload = value.model_dump(mode="json") | {"reconciliation_id": "pending"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        value = value.model_copy(update={"reconciliation_id": f"reconciliation-{digest}"})
        self.append("forecast_reconciliations.jsonl", value)
        return value

    def run(
        self,
        day: date,
        shortlist: list[str],
        priors: dict[str, float],
        signals: pd.DataFrame,
        matured_feedback: list[dict[str, Any]],
    ) -> ForecastingResult:
        analyses: dict[str, CompanyAnalysis] = {}
        forecasts: dict[str, StockForecast] = {}
        for symbol in shortlist:
            analysis, forecast = self.research.analyze_and_forecast(
                day, symbol, float(priors[symbol])
            )
            analyses[symbol] = analysis
            forecasts[symbol] = forecast
            self.append("company_analysis.jsonl", analysis)
            self.append("stock_forecasts.jsonl", forecast)

        tools = ForecastingTools(self.store, signals)
        observations: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        signatures: set[tuple[str, tuple[str, ...], str]] = set()
        revisions = {symbol: 0 for symbol in shortlist}
        reconciliation: ForecastReconciliation | None = None
        reconciliation_version = 0

        for step in range(self.reasoning.max_steps):
            decision = self._decide(
                day, step, shortlist, analyses, forecasts, observations, reconciliation
            )
            action_record = {
                "month_id": day.isoformat(),
                "step": step,
                **decision.model_dump(mode="json"),
                "status": "selected",
            }
            signature_question = (
                decision.question.strip().lower()
                if decision.action == ForecastAction.REVISE_FORECASTS
                else ""
            )
            signature = (decision.action.value, decision.symbols, signature_question)
            if signature in signatures:
                action_record["status"] = "stopped_duplicate_action"
                actions.append(action_record)
                self.append("agent_actions.jsonl", action_record)
                break
            signatures.add(signature)
            actions.append(action_record)
            self.append("agent_actions.jsonl", action_record)

            observation: dict[str, Any] | None = None
            if decision.action == ForecastAction.INSPECT_MARKET_REGIME:
                observation = tools.market_regime(day)
            elif decision.action == ForecastAction.INSPECT_PEER_CONTEXT:
                observation = tools.peer_context(decision.symbols)
            elif decision.action == ForecastAction.INSPECT_COMPANY_HISTORY:
                observation = tools.company_history(day, decision.symbols)
            elif decision.action == ForecastAction.INSPECT_CALIBRATION:
                observation = tools.calibration(matured_feedback)
            elif decision.action == ForecastAction.REVISE_FORECASTS:
                revised: dict[str, Any] = {}
                for symbol in decision.symbols:
                    revisions[symbol] += 1
                    focus = {
                        "question": decision.question,
                        "prior_analysis": analyses[symbol].model_dump(mode="json"),
                        "prior_forecast": forecasts[symbol].model_dump(mode="json"),
                        "tool_observations": observations,
                        "instruction": (
                            "Reconsider the forecast only if the new evidence changes the "
                            "estimate; explicitly preserve uncertainty and avoid false precision."
                        ),
                    }
                    analysis, forecast = self.research.analyze_and_forecast(
                        day,
                        symbol,
                        float(priors[symbol]),
                        focus_context=focus,
                        revision=revisions[symbol],
                    )
                    analyses[symbol] = analysis
                    forecasts[symbol] = forecast
                    self.append("company_analysis.jsonl", analysis)
                    self.append("stock_forecasts.jsonl", forecast)
                    revised[symbol] = {
                        "analysis_id": analysis.analysis_id,
                        "forecast_id": forecast.forecast_id,
                        "expected_return_3m": forecast.expected_return_3m,
                        "confidence": forecast.confidence,
                    }
                observation = {"revised_forecasts": revised}
                reconciliation = None
            elif decision.action == ForecastAction.RECONCILE_FORECASTS:
                reconciliation_version += 1
                reconciliation = self._reconcile(
                    day,
                    analyses,
                    forecasts,
                    priors,
                    observations,
                    reconciliation_version,
                )
                observation = {
                    "reconciliation_id": reconciliation.reconciliation_id,
                    "forecast_count": len(reconciliation.forecasts),
                    "confidence": reconciliation.confidence,
                }
            elif decision.action == ForecastAction.FINISH:
                if reconciliation is not None:
                    break
                observation = {
                    "status": "finish_rejected",
                    "reason": "all forecasts must be reconciled before finishing",
                }

            if observation is not None:
                observation_record = {
                    "observation_id": f"observation-{day}-{step}",
                    "month_id": day.isoformat(),
                    "step": step,
                    "tool": decision.action.value,
                    "symbols": decision.symbols,
                    "question": decision.question,
                    "result": observation,
                }
                observations.append(observation_record)
                self.append("agent_observations.jsonl", observation_record)
            if (
                reconciliation is not None
                and decision.expected_information_gain < self.reasoning.min_information_gain
            ):
                break

        if reconciliation is None:
            reconciliation_version += 1
            reconciliation = self._reconcile(
                day,
                analyses,
                forecasts,
                priors,
                observations,
                reconciliation_version,
            )
            forced = {
                "month_id": day.isoformat(),
                "step": len(actions),
                "action": ForecastAction.RECONCILE_FORECASTS.value,
                "symbols": [],
                "question": "Produce the required final cross-company forecast set.",
                "rationale": "The reasoning budget ended before a final reconciliation.",
                "expected_information_gain": 0.0,
                "status": "forced_safety_completion",
            }
            actions.append(forced)
            self.append("agent_actions.jsonl", forced)

        return ForecastingResult(
            analyses=analyses,
            forecasts=forecasts,
            reconciliation=reconciliation,
            actions=tuple(actions),
            observations=tuple(observations),
        )

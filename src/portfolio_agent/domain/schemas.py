"""Canonical public contracts shared with the independent data workstream."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"
Weight = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Provenance(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    RETROSPECTIVE = "retrospective"


class UniverseRecord(StrictModel):
    symbol: str = Field(min_length=1, max_length=24)
    company_name: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    eligible_from: date
    eligible_until: date | None = None
    universe_formation_date: date
    source_id: str = Field(min_length=1)
    available_at: datetime
    currency: str = Field(pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def validate_dates(self) -> UniverseRecord:
        if self.eligible_until is not None and self.eligible_until <= self.eligible_from:
            raise ValueError("eligible_until must be after eligible_from")
        if self.universe_formation_date > self.eligible_from:
            raise ValueError("formation date cannot follow eligibility start")
        return self


class PriceRecord(StrictModel):
    date: date
    symbol: str = Field(min_length=1)
    adjusted_close: FiniteFloat = Field(gt=0)
    available_at: datetime
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    adjustment_method: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class FactorRecord(StrictModel):
    date: date
    factor_name: str = Field(min_length=1)
    value: FiniteFloat
    available_at: datetime
    source_id: str = Field(min_length=1)
    unit: str | None = None
    frequency: str | None = None


class ArticleRecord(StrictModel):
    record_id: str = Field(min_length=1)
    symbols: tuple[str, ...]
    title: str = Field(min_length=1)
    summary: str | None = None
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    event_type: str | None = None
    published_at: datetime
    available_at: datetime
    license: str = Field(min_length=1)
    is_synthetic: bool
    provenance: str = Field(min_length=1)
    event_id: str | None = None
    exploratory_only: bool = False

    @field_validator("symbols")
    @classmethod
    def symbols_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("symbols must be non-empty and unique")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> ArticleRecord:
        if self.available_at < self.published_at:
            raise ValueError("available_at cannot precede published_at")
        return self


class FundamentalRecord(StrictModel):
    symbol: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: FiniteFloat
    unit: str = Field(min_length=1)
    period_end: date
    filing_published_at: datetime
    available_at: datetime
    source_id: str = Field(min_length=1)
    data_revision: str = Field(min_length=1)
    retrospective_only: bool = False

    @model_validator(mode="after")
    def validate_availability(self) -> FundamentalRecord:
        if self.available_at < self.filing_published_at:
            raise ValueError("available_at cannot precede filing publication")
        if self.filing_published_at.date() < self.period_end:
            raise ValueError("filing publication cannot precede period end")
        return self


class Holding(StrictModel):
    symbol: str = Field(min_length=1)
    weight: Weight


class PortfolioConstraints(StrictModel):
    holding_count: int = Field(default=10, ge=1)
    min_weight: Weight = 0.02
    max_weight: Weight = 0.25
    max_one_way_turnover: Weight = 0.20
    long_only: bool = True
    allow_leverage: bool = False


class InitialPortfolio(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    portfolio_id: str = Field(min_length=1)
    as_of: date
    base_currency: str = Field(pattern=r"^[A-Z]{3}$")
    benchmark: str = Field(min_length=1)
    initial_nav: FiniteFloat = Field(gt=0)
    holdings: tuple[Holding, ...]
    constraints: PortfolioConstraints
    is_synthetic: bool
    data_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_holdings(self) -> InitialPortfolio:
        symbols = [holding.symbol for holding in self.holdings]
        if len(symbols) != self.constraints.holding_count or len(set(symbols)) != len(symbols):
            raise ValueError("holdings must contain the configured number of distinct symbols")
        if abs(sum(holding.weight for holding in self.holdings) - 1.0) > 1e-6:
            raise ValueError("holding weights must sum to one")
        if any(
            not self.constraints.min_weight <= holding.weight <= self.constraints.max_weight
            for holding in self.holdings
        ):
            raise ValueError("holding weight outside configured bounds")
        return self


class CompanyAnalysis(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    analysis_id: str
    decision_as_of: datetime
    created_at: datetime
    symbol: str
    provider: str
    model: str
    model_settings_hash: str
    prompt_version: str
    evidence_ids: tuple[str, ...]
    bull_drivers: tuple[str, ...]
    bear_drivers: tuple[str, ...]
    key_catalysts: tuple[str, ...]
    risks: tuple[str, ...]
    confidence: Confidence
    rationale: str
    quality_flags: tuple[str, ...] = ()
    retrospective_model: bool


class StockForecast(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    forecast_id: str
    decision_as_of: datetime
    created_at: datetime
    symbol: str
    horizon_sessions: Literal[63] = 63
    expected_return_3m: FiniteFloat = Field(ge=-1.0, le=3.0)
    confidence: Confidence
    rationale: str
    analysis_id: str
    evidence_ids: tuple[str, ...]
    provider: str
    model: str
    model_settings_hash: str
    prompt_version: str
    is_retrospective: bool


class ForecastAction(StrEnum):
    """Bounded actions available to the monthly forecasting agent."""

    EXECUTE_SKILL = "execute_skill"
    INSPECT_MARKET_REGIME = "inspect_market_regime"
    INSPECT_PEER_CONTEXT = "inspect_peer_context"
    INSPECT_COMPANY_HISTORY = "inspect_company_history"
    INSPECT_CALIBRATION = "inspect_calibration"
    REVISE_FORECASTS = "revise_forecasts"
    RECONCILE_FORECASTS = "reconcile_forecasts"
    FINISH = "finish"


class ForecastAgentDecision(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    action: ForecastAction
    skill_name: str | None = None
    symbols: tuple[str, ...] = ()
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_information_gain: Confidence

    @model_validator(mode="after")
    def validate_targets(self) -> ForecastAgentDecision:
        targeted = {
            ForecastAction.INSPECT_PEER_CONTEXT,
            ForecastAction.INSPECT_COMPANY_HISTORY,
            ForecastAction.REVISE_FORECASTS,
        }
        if self.action in targeted and not self.symbols:
            raise ValueError(f"{self.action} requires at least one symbol")
        if len(self.symbols) > 5 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique and contain at most five targets")
        if self.action == ForecastAction.EXECUTE_SKILL and not self.skill_name:
            raise ValueError("execute_skill requires skill_name")
        if self.action != ForecastAction.EXECUTE_SKILL and self.skill_name is not None:
            raise ValueError("skill_name is accepted only for execute_skill")
        if self.action in {
            ForecastAction.INSPECT_MARKET_REGIME,
            ForecastAction.INSPECT_CALIBRATION,
            ForecastAction.RECONCILE_FORECASTS,
            ForecastAction.FINISH,
        } and self.symbols:
            raise ValueError(f"{self.action} does not accept symbol targets")
        return self


class ReconciledStockForecast(StrictModel):
    symbol: str = Field(min_length=1)
    source_forecast_id: str = Field(min_length=1)
    raw_expected_return_3m: FiniteFloat = Field(ge=-1.0, le=3.0)
    deterministic_expected_return_3m: FiniteFloat | None = Field(
        default=None, ge=-1.0, le=3.0
    )
    agent_expected_return_3m: FiniteFloat | None = Field(default=None, ge=-1.0, le=3.0)
    agent_blend_weight: Weight | None = None
    expected_return_3m: FiniteFloat = Field(ge=-1.0, le=3.0)
    confidence: Confidence
    relative_rank: int = Field(ge=1)
    bear_case_return: FiniteFloat = Field(ge=-1.0, le=3.0)
    bull_case_return: FiniteFloat = Field(ge=-1.0, le=3.0)
    adjustment_rationale: str = Field(min_length=1)
    consistency_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_scenario_order(self) -> ReconciledStockForecast:
        if not self.bear_case_return <= self.expected_return_3m <= self.bull_case_return:
            raise ValueError("expected return must lie between bear and bull cases")
        return self


class ForecastReconciliation(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    reconciliation_id: str = "pending"
    decision_as_of: datetime
    provider: str = "pending"
    model: str = "pending"
    market_regime: str = Field(min_length=1)
    market_expected_return_3m: FiniteFloat = Field(ge=-1.0, le=3.0)
    common_factor_assumptions: tuple[str, ...]
    forecasts: tuple[ReconciledStockForecast, ...]
    cross_sectional_risks: tuple[str, ...]
    confidence: Confidence
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cross_section(self) -> ForecastReconciliation:
        symbols = [forecast.symbol for forecast in self.forecasts]
        ranks = [forecast.relative_rank for forecast in self.forecasts]
        if not symbols or len(symbols) != len(set(symbols)):
            raise ValueError("reconciliation forecasts must contain unique symbols")
        if set(ranks) != set(range(1, len(ranks) + 1)):
            raise ValueError("relative ranks must be contiguous from one")
        return self


class ExperimentType(StrEnum):
    RESEARCH_BLEND = "research_blend"
    FORECAST_CONFIDENCE_SCALE = "forecast_confidence_scale"
    RISK_AVERSION = "risk_aversion"
    TURNOVER_PENALTY = "turnover_penalty"
    COVARIANCE_LOOKBACK = "covariance_lookback"


class ExperimentProposal(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    experiment_type: ExperimentType
    hypothesis: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    diagnosis_ids: tuple[str, ...] = ()
    overrides: dict[str, float | int]
    expected_tradeoffs: tuple[str, ...]

    @model_validator(mode="after")
    def validate_allowlist(self) -> ExperimentProposal:
        rules: dict[ExperimentType, tuple[str, set[int] | tuple[float, float]]] = {
            ExperimentType.RESEARCH_BLEND: ("research_blend", (0.0, 1.0)),
            ExperimentType.FORECAST_CONFIDENCE_SCALE: ("confidence_scale", (0.25, 1.0)),
            ExperimentType.RISK_AVERSION: ("risk_aversion", (0.5, 8.0)),
            ExperimentType.TURNOVER_PENALTY: ("turnover_penalty", (0.0, 5.0)),
            ExperimentType.COVARIANCE_LOOKBACK: (
                "covariance_lookback_sessions",
                {126, 189, 252, 378, 504},
            ),
        }
        key, allowed = rules[self.experiment_type]
        if set(self.overrides) != {key}:
            raise ValueError(f"experiment must override only {key}")
        value = self.overrides[key]
        if isinstance(allowed, set):
            if value not in allowed:
                raise ValueError(f"{key} is not an allowed value")
        elif not allowed[0] <= float(value) <= allowed[1]:
            raise ValueError(f"{key} is outside its allowed range")
        return self


class Projection(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    projection_id: str
    run_id: str
    month_id: str
    experiment_id: str | None = None
    decision_as_of: datetime
    generated_at: datetime
    horizon_sessions: Literal[63] = 63
    strategy_version: str
    weights: tuple[Holding, ...]
    expected_return_3m: FiniteFloat
    annualized_volatility: FiniteFloat = Field(gt=0)
    model_assumptions: dict[str, str | int | float]
    seed: int
    forecast_ids: tuple[str, ...]
    p10: tuple[tuple[date, float], ...]
    p50: tuple[tuple[date, float], ...]
    p90: tuple[tuple[date, float], ...]
    initial_nav: FiniteFloat = Field(gt=0)
    content_hash: str
    is_frozen: Literal[True] = True


class ForecastOutcome(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    projection_id: str
    matured_at: datetime
    realized_63_session_return: FiniteFloat
    forecast_error: FiniteFloat
    coverage: Confidence
    status: Literal["matured", "partial", "unavailable"]


class DatasetManifest(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    dataset_version: str
    profile: Literal["smoke", "historical", "synthetic-full"]
    generated_at: datetime
    is_synthetic: bool
    files: dict[str, str]
    coverage: dict[str, int | str | float]
    rights_summary_path: str

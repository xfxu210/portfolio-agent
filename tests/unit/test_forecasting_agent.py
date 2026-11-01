import json
from datetime import date
from pathlib import Path

from portfolio_agent.agents.forecasting import ForecastingAgent, ForecastingTools
from portfolio_agent.agents.providers.mock import MockProvider
from portfolio_agent.agents.research import ResearchService
from portfolio_agent.config import ReasoningConfig
from portfolio_agent.data import DatasetStore
from portfolio_agent.persistence import ArtifactWriter
from portfolio_agent.portfolio.signals import build_signal_table


class CapturingMockProvider(MockProvider):
    def __init__(self) -> None:
        self.contexts: dict[str, dict] = {}

    def generate_structured(self, **kwargs):
        self.contexts[kwargs["task"]] = json.loads(kwargs["messages"][-1]["content"])
        return super().generate_structured(**kwargs)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_forecasting_agent_dynamically_researches_and_reconciles_every_symbol(tmp_path) -> None:
    store = DatasetStore.load(Path("examples/smoke"))
    provider = MockProvider()
    writer = ArtifactWriter(tmp_path)
    research = ResearchService(
        store,
        provider,
        trace=lambda value: writer.append("traces.jsonl", value),
    )
    signals = build_signal_table(store, date(2025, 1, 2), research_blend=0.5)
    shortlist = signals["symbol"].head(10).tolist()
    priors = {
        symbol: float(value) * 0.25
        for symbol, value in signals.set_index("symbol").loc[shortlist, "momentum"].items()
    }
    agent = ForecastingAgent(
        store,
        provider,
        research,
        ReasoningConfig(),
        writer.append,
    )

    result = agent.run(date(2025, 1, 2), shortlist, priors, signals, matured_feedback=[])

    assert {forecast.symbol for forecast in result.reconciliation.forecasts} == set(shortlist)
    assert [forecast.relative_rank for forecast in result.reconciliation.forecasts] == list(
        range(1, len(shortlist) + 1)
    )
    assert all(
        forecast.deterministic_expected_return_3m is not None
        and forecast.agent_expected_return_3m is not None
        and 0.25 <= (forecast.agent_blend_weight or 0) <= 0.75
        and "runtime_hybrid_blend" in forecast.consistency_flags
        for forecast in result.reconciliation.forecasts
    )
    actions = [item["action"] for item in result.actions]
    assert actions == [
        "execute_skill",
        "execute_skill",
        "execute_skill",
        "execute_skill",
        "revise_forecasts",
        "reconcile_forecasts",
        "finish",
    ]
    assert [item["skill_name"] for item in result.actions[:4]] == [
        "market-regime",
        "company-deep-dive",
        "forecast-calibration",
        "peer-relative-value",
    ]
    assert len(_read_jsonl(tmp_path / "forecast_reconciliations.jsonl")) == 1
    assert len(_read_jsonl(tmp_path / "agent_observations.jsonl")) == 6
    assert len(_read_jsonl(tmp_path / "skill_executions.jsonl")) == 4
    assert len(_read_jsonl(tmp_path / "deterministic_inferences.jsonl")) == len(shortlist)
    assert len(_read_jsonl(tmp_path / "stock_forecasts.jsonl")) == len(shortlist) + 2
    assert set(result.deterministic_inferences) == set(shortlist)
    assert all(
        inference["data_coverage"] == 1.0
        for inference in result.deterministic_inferences.values()
    )


def test_research_forecast_receives_the_generated_company_analysis(tmp_path) -> None:
    store = DatasetStore.load(Path("examples/smoke"))
    provider = CapturingMockProvider()
    service = ResearchService(
        store,
        provider,
        trace=lambda value: ArtifactWriter(tmp_path).append("traces.jsonl", value),
    )

    analysis, forecast = service.analyze_and_forecast(
        date(2025, 1, 2), "SMOKE01", 0.02
    )

    assert forecast.analysis_id == analysis.analysis_id
    assert analysis.analysis_id.startswith("analysis-smoke01-")
    assert forecast.forecast_id.startswith("forecast-smoke01-")
    assert provider.contexts["stock_forecast"]["company_analysis"]["analysis_id"] == (
        analysis.analysis_id
    )


def test_skill_tools_use_news_fundamentals_market_prices_and_portfolio() -> None:
    store = DatasetStore.load(Path("examples/smoke"))
    signals = build_signal_table(store, date(2025, 1, 2), research_blend=0.5)
    tools = ForecastingTools(store, signals)
    symbols = tuple(signals["symbol"].head(3))
    priors = {
        symbol: float(signals.set_index("symbol").loc[symbol, "momentum"]) * 0.25
        for symbol in symbols
    }

    inference = tools.deterministic_inference(date(2025, 1, 2), symbols, priors)
    portfolio = tools.portfolio_context(symbols, {})

    assert set(inference) == set(symbols)
    assert all(set(item["data_channels"]) == {
        "prices",
        "news",
        "fundamentals",
        "market_and_factors",
    } for item in inference.values())
    assert portfolio["baseline_holdings"]
    assert set(portfolio["targets"]) == set(symbols)


def test_calibration_learns_regularized_mapping_from_matured_forecasts_only() -> None:
    feedback = [
        {
            "symbol": f"S{index}",
            "raw_expected_return_3m": expected,
            "expected_return_3m": expected,
            "realized_63_session_return": 0.01 - 0.5 * expected,
            "forecast_error": 0.01 - 1.5 * expected,
        }
        for index, expected in enumerate([value / 100 for value in range(-20, 20)])
    ]

    calibration = ForecastingTools.calibration(feedback)

    assert calibration["matured_forecasts"] == 40
    assert calibration["calibration_accepted"] is True
    assert calibration["standalone_calibration_slope"] < 0
    assert calibration["calibration_reliability"] == 0.2

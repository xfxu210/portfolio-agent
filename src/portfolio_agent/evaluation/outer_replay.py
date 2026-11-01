from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_agent.agents.providers.base import LLMProvider
from portfolio_agent.agents.research import ResearchService
from portfolio_agent.config import ReplayConfig
from portfolio_agent.data.loader import DatasetStore
from portfolio_agent.domain.schemas import ExperimentProposal
from portfolio_agent.evaluation.analytics import performance_metrics
from portfolio_agent.evaluation.inner_walkforward import trailing_walkforward_score
from portfolio_agent.persistence import ArtifactWriter, SQLiteLedger
from portfolio_agent.persistence.artifact_writer import content_hash, sha256_path
from portfolio_agent.portfolio.accounting import PortfolioAccount, price_map
from portfolio_agent.portfolio.optimizer import optimize_portfolio, select_ten
from portfolio_agent.portfolio.signals import build_signal_table
from portfolio_agent.projections.monte_carlo import projection_fan


@dataclass
class StrategyParams:
    research_blend: float = 0.5
    confidence_scale: float = 0.75
    risk_aversion: float = 2.0
    turnover_penalty: float = 0.5
    covariance_lookback_sessions: int = 126

    def apply(self, proposal: ExperimentProposal) -> StrategyParams:
        values = self.__dict__ | proposal.overrides
        return StrategyParams(**values)

    @property
    def version(self) -> str:
        return hashlib.sha256(str(sorted(self.__dict__.items())).encode()).hexdigest()[:12]


@dataclass
class Candidate:
    proposal: ExperimentProposal
    params: StrategyParams
    weights: dict[str, float]
    score: float
    turnover: float
    expected_return_3m: float
    annualized_volatility: float
    projection: dict[str, Any]


def _returns_matrix(store: DatasetStore, day: date, lookback: int) -> pd.DataFrame:
    history = store.price_history(day)
    pivot = history.pivot(index="date", columns="symbol", values="adjusted_close")
    return pivot.pct_change().dropna().tail(lookback)


def _future_sessions(store: DatasetStore, day: date) -> list[date]:
    all_sessions = sorted(store.prices["date"].unique())
    later = [value for value in all_sessions if value > day][:63]
    if len(later) < 63:
        start = pd.Timestamp(later[-1] if later else day)
        extra = pd.bdate_range(start + pd.Timedelta(days=1), periods=63 - len(later))
        later.extend(value.date() for value in extra)
    return later


class ReplayEngine:
    def __init__(
        self,
        store: DatasetStore,
        provider: LLMProvider,
        config: ReplayConfig,
        output: Path,
    ) -> None:
        self.store = store
        self.provider = provider
        self.config = config
        self.output = output
        self.writer = ArtifactWriter(output)
        self.research = ResearchService(
            store,
            provider,
            trace=lambda value: self.writer.append("traces.jsonl", value),
        )
        self.run_id = hashlib.sha256(
            f"{store.dataset_hash}:{config.seed}:{config.agent.mode}".encode()
        ).hexdigest()[:16]

    def _proposal(
        self,
        day: date,
        iteration: int,
        evidence_ids: list[str],
        matured_feedback: list[dict[str, Any]],
        within_month_feedback: list[dict[str, Any]],
    ) -> ExperimentProposal:
        context = {
            "decision_as_of": datetime.combine(day, datetime.max.time(), tzinfo=UTC).isoformat(),
            "iteration": iteration,
            "evidence_ids": evidence_ids,
            "matured_feedback": matured_feedback[-10:],
            "within_month_feedback": within_month_feedback,
        }
        result = self.provider.generate_structured(
            task="experiment_proposal",
            messages=[{"role": "user", "content": json.dumps(context)}],
            response_model=ExperimentProposal,
            request_id=f"proposal:{day}:{iteration}",
        )
        self.writer.append(
            "traces.jsonl",
            {
                "stage": "experiment_proposal",
                "request_id": f"proposal:{day}:{iteration}",
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
        return result.value

    def _realized_outcome(self, projection: dict[str, Any]) -> dict[str, Any] | None:
        sessions = sorted(self.store.prices["date"].unique())
        decision_day = date.fromisoformat(str(projection["month_id"]))
        try:
            decision_index = sessions.index(decision_day)
        except ValueError:
            return None
        execution_index = decision_index + 1
        mature_index = execution_index + 63
        if mature_index >= len(sessions):
            return None
        start_prices = price_map(self.store.prices, sessions[execution_index])
        end_prices = price_map(self.store.prices, sessions[mature_index])
        weights = projection["weights"]
        covered = {
            symbol: weight
            for symbol, weight in weights.items()
            if symbol in start_prices and symbol in end_prices
        }
        coverage = sum(covered.values())
        if coverage <= 0:
            return None
        realized = sum(
            weight / coverage * (end_prices[symbol] / start_prices[symbol] - 1)
            for symbol, weight in covered.items()
        )
        return {
            "projection_id": projection["projection_id"],
            "matured_at": datetime.combine(
                sessions[mature_index], datetime.max.time(), tzinfo=UTC
            ).isoformat(),
            "realized_63_session_return": realized,
            "forecast_error": realized - projection["expected_return_3m"],
            "coverage": coverage,
            "status": "matured" if coverage >= 1 - 1e-6 else "partial",
        }

    def _candidate(
        self,
        day: date,
        previous: dict[str, float],
        params: StrategyParams,
        proposal: ExperimentProposal,
        forecast_returns: dict[str, float],
        matured_feedback: list[dict[str, Any]],
        experiment_id: str,
    ) -> Candidate | None:
        candidate_params = params.apply(proposal)
        signals = build_signal_table(
            self.store, day, research_blend=candidate_params.research_blend
        )
        selected = select_ten(signals, previous)
        returns = _returns_matrix(self.store, day, candidate_params.covariance_lookback_sessions)
        covariance = returns[selected].cov() * 252
        priors = signals.set_index("symbol")["momentum"] * 0.25
        expected = pd.Series(
            {
                symbol: candidate_params.confidence_scale
                * forecast_returns.get(symbol, float(priors[symbol]))
                + (1 - candidate_params.confidence_scale) * float(priors[symbol])
                for symbol in selected
            }
        )
        optimized = optimize_portfolio(
            expected,
            covariance,
            previous,
            risk_aversion=candidate_params.risk_aversion,
            turnover_penalty=candidate_params.turnover_penalty,
        )
        if optimized.status != "optimal":
            self.writer.append(
                "experiments.jsonl",
                {
                    "experiment_id": experiment_id,
                    "month_id": day.isoformat(),
                    "status": optimized.status,
                    "reason": optimized.reason,
                    "proposal": proposal,
                },
            )
            return None
        weight_vector = pd.Series(optimized.weights)
        calibration_error = (
            float(np.mean([abs(item["forecast_error"]) for item in matured_feedback[-10:]]))
            if matured_feedback
            else 0.0
        )
        score = trailing_walkforward_score(
            returns,
            weight_vector,
            risk_aversion=candidate_params.risk_aversion,
            one_way_turnover=optimized.one_way_turnover,
            fee_bps=self.config.fee_bps,
            calibration_error=calibration_error,
            confidence_scale=candidate_params.confidence_scale,
        )
        expected_return = float(expected.dot(weight_vector))
        volatility = float(np.sqrt(weight_vector @ covariance @ weight_vector))
        future = _future_sessions(self.store, day)
        fan = projection_fan(
            initial_nav=100.0,
            expected_return_3m=expected_return,
            annualized_volatility=max(volatility, 1e-6),
            future_sessions=future,
            seed=self.config.seed + int(day.strftime("%j")) + len(experiment_id),
        )
        projection_base = {
            "run_id": self.run_id,
            "month_id": day.isoformat(),
            "experiment_id": experiment_id,
            "decision_as_of": datetime.combine(day, datetime.max.time(), tzinfo=UTC),
            "horizon_sessions": 63,
            "strategy_version": candidate_params.version,
            "weights": optimized.weights,
            "expected_return_3m": expected_return,
            "annualized_volatility": volatility,
            "seed": self.config.seed,
            **fan,
            "initial_nav": 100.0,
            "is_frozen": True,
        }
        projection = {
            **projection_base,
            "projection_id": f"projection-{content_hash(projection_base)[:16]}",
            "content_hash": content_hash(projection_base),
        }
        self.writer.append("projections.jsonl", projection)
        return Candidate(
            proposal=proposal,
            params=candidate_params,
            weights=optimized.weights,
            score=score,
            turnover=optimized.one_way_turnover,
            expected_return_3m=expected_return,
            annualized_volatility=volatility,
            projection=projection,
        )

    def _adaptive_schedule(
        self, dates: list[date]
    ) -> tuple[dict[date, dict[str, float]], list[dict[str, Any]], StrategyParams]:
        previous = {
            holding.symbol: holding.weight for holding in self.store.initial_portfolio.holdings
        }
        params = StrategyParams()
        schedule: dict[date, dict[str, float]] = {}
        accepted_history: list[dict[str, Any]] = []
        accepted_projections: list[dict[str, Any]] = []
        matured_feedback: list[dict[str, Any]] = []
        matured_ids: set[str] = set()
        development_scores: dict[str, list[float]] = {}
        development_params: dict[str, StrategyParams] = {}
        for month_index, day in enumerate(dates):
            for pending_projection in accepted_projections:
                projection_id = pending_projection["projection_id"]
                if projection_id in matured_ids:
                    continue
                outcome = self._realized_outcome(pending_projection)
                if outcome is None or outcome["matured_at"][:10] > day.isoformat():
                    continue
                matured_feedback.append(outcome)
                matured_ids.add(projection_id)
            signals = build_signal_table(self.store, day, research_blend=params.research_blend)
            shortlist = list(dict.fromkeys([*previous, *signals["symbol"].head(15)]))[:20]
            analyses = []
            forecasts = []
            forecast_returns: dict[str, float] = {}
            priors = signals.set_index("symbol")["momentum"] * 0.25
            for symbol in shortlist:
                analysis, forecast = self.research.analyze_and_forecast(
                    day, symbol, float(priors[symbol])
                )
                analyses.append(analysis)
                forecasts.append(forecast)
                forecast_returns[symbol] = forecast.expected_return_3m
                self.writer.append("company_analysis.jsonl", analysis)
                self.writer.append("stock_forecasts.jsonl", forecast)
            evidence_ids = sorted(
                {evidence_id for analysis in analyses for evidence_id in analysis.evidence_ids}
            )[-30:]
            candidates: list[Candidate] = []
            within_month_feedback: list[dict[str, Any]] = []
            for iteration in range(self.config.iterations):
                proposal = self._proposal(
                    day,
                    iteration,
                    evidence_ids,
                    matured_feedback,
                    within_month_feedback,
                )
                experiment_id = f"experiment-{day}-{iteration}"
                candidate = self._candidate(
                    day,
                    previous,
                    params,
                    proposal,
                    forecast_returns,
                    matured_feedback,
                    experiment_id,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    within_month_feedback.append(
                        {
                            "iteration": iteration,
                            "overrides": proposal.overrides,
                            "status": "feasible",
                            "score": candidate.score,
                            "turnover": candidate.turnover,
                        }
                    )
                    if month_index < int(len(dates) * self.config.development_fraction):
                        development_scores.setdefault(candidate.params.version, []).append(
                            candidate.score
                        )
                        development_params[candidate.params.version] = candidate.params
                else:
                    within_month_feedback.append(
                        {
                            "iteration": iteration,
                            "overrides": proposal.overrides,
                            "status": "infeasible",
                        }
                    )
            if not candidates:
                schedule[day] = previous
                returns = _returns_matrix(self.store, day, params.covariance_lookback_sessions)
                available = [symbol for symbol in previous if symbol in returns.columns]
                weights = pd.Series({symbol: previous[symbol] for symbol in available})
                weights /= weights.sum()
                trailing = returns[available].dot(weights)
                expected_return = float(trailing.mean() * 63)
                volatility = float(trailing.std(ddof=0) * np.sqrt(252))
                fan = projection_fan(
                    initial_nav=100.0,
                    expected_return_3m=expected_return,
                    annualized_volatility=max(volatility, 1e-6),
                    future_sessions=_future_sessions(self.store, day),
                    seed=self.config.seed + int(day.strftime("%j")),
                )
                projection_base = {
                    "run_id": self.run_id,
                    "month_id": day.isoformat(),
                    "experiment_id": None,
                    "decision_as_of": datetime.combine(day, datetime.max.time(), tzinfo=UTC),
                    "horizon_sessions": 63,
                    "strategy_version": params.version,
                    "weights": previous,
                    "expected_return_3m": expected_return,
                    "annualized_volatility": volatility,
                    "seed": self.config.seed,
                    **fan,
                    "initial_nav": 100.0,
                    "is_frozen": True,
                }
                projection = {
                    **projection_base,
                    "projection_id": f"projection-{content_hash(projection_base)[:16]}",
                    "content_hash": content_hash(projection_base),
                }
                self.writer.append("projections.jsonl", projection)
                accepted_projections.append(projection)
                decision = {
                    "run_id": self.run_id,
                    "month_id": day.isoformat(),
                    "decision_as_of": day.isoformat(),
                    "strategy_version": params.version,
                    "strategy_params": params.__dict__,
                    "weights": previous,
                    "accepted_experiment_id": None,
                    "accepted_projection_id": projection["projection_id"],
                    "decision_status": "incumbent_retained_no_feasible_candidate",
                    "provider": self.provider.provider,
                    "model": self.provider.model,
                    "development": month_index < int(len(dates) * self.config.development_fraction),
                }
                accepted_history.append(decision)
                self.writer.append("monthly_decisions.jsonl", decision)
                self.writer.append("accepted_allocations.jsonl", decision)
                continue
            winner = max(candidates, key=lambda candidate: candidate.score)
            for candidate in candidates:
                accepted = candidate is winner
                self.writer.append(
                    "experiments.jsonl",
                    {
                        "experiment_id": candidate.projection["experiment_id"],
                        "month_id": day.isoformat(),
                        "status": "accepted" if accepted else "rejected",
                        "proposal": candidate.proposal,
                        "strategy_version": candidate.params.version,
                        "candidate_weights": candidate.weights,
                        "inner_validation_score": candidate.score,
                        "one_way_turnover": candidate.turnover,
                        "projection_id": candidate.projection["projection_id"],
                    },
                )
            params = winner.params
            previous = winner.weights
            accepted_projections.append(winner.projection)
            schedule[day] = previous
            decision = {
                "run_id": self.run_id,
                "month_id": day.isoformat(),
                "decision_as_of": day.isoformat(),
                "strategy_version": params.version,
                "strategy_params": params.__dict__,
                "weights": previous,
                "accepted_experiment_id": winner.projection["experiment_id"],
                "accepted_projection_id": winner.projection["projection_id"],
                "decision_status": "candidate_accepted",
                "provider": self.provider.provider,
                "model": self.provider.model,
                "development": month_index < int(len(dates) * self.config.development_fraction),
            }
            accepted_history.append(decision)
            self.writer.append("monthly_decisions.jsonl", decision)
            self.writer.append("accepted_allocations.jsonl", decision)
        if development_scores:
            selected_version = max(
                development_scores,
                key=lambda version: float(np.mean(development_scores[version])),
            )
            fixed_params = development_params[selected_version]
        else:
            fixed_params = StrategyParams()
        return schedule, accepted_history, fixed_params

    def _fixed_schedule(
        self, dates: list[date], params: StrategyParams
    ) -> dict[date, dict[str, float]]:
        previous = {
            holding.symbol: holding.weight for holding in self.store.initial_portfolio.holdings
        }
        schedule = {}
        for day in dates:
            signals = build_signal_table(self.store, day, research_blend=params.research_blend)
            selected = select_ten(signals, previous)
            returns = _returns_matrix(self.store, day, params.covariance_lookback_sessions)
            expected = signals.set_index("symbol").loc[selected, "momentum"] * 0.25
            result = optimize_portfolio(
                expected,
                returns[selected].cov() * 252,
                previous,
                risk_aversion=params.risk_aversion,
                turnover_penalty=params.turnover_penalty,
            )
            if result.status == "optimal":
                previous = result.weights
            schedule[day] = previous
        return schedule

    def _simulate(self, name: str, schedule: dict[date, dict[str, float]]) -> pd.DataFrame:
        sessions = sorted(self.store.prices["date"].unique())
        initial = {
            holding.symbol: holding.weight for holding in self.store.initial_portfolio.holdings
        }
        first_prices = price_map(self.store.prices, sessions[0])
        account = PortfolioAccount.from_weights(
            self.store.initial_portfolio.initial_nav,
            initial,
            first_prices,
            self.config.fee_bps,
        )
        pending: dict[str, float] | None = None
        rows = []
        for day in sessions:
            current_prices = price_map(self.store.prices, day)
            if pending is not None:
                account.rebalance(pending, current_prices)
                pending = None
            rows.append({"date": day, "strategy": name, "nav": account.value(current_prices)})
            if day in schedule:
                pending = schedule[day]
        return pd.DataFrame(rows)

    def _average_turnover(self, schedule: dict[date, dict[str, float]]) -> float:
        previous = {
            holding.symbol: holding.weight for holding in self.store.initial_portfolio.holdings
        }
        values = []
        for day in sorted(schedule):
            target = schedule[day]
            values.append(
                0.5
                * sum(
                    abs(target.get(symbol, 0.0) - previous.get(symbol, 0.0))
                    for symbol in set(target) | set(previous)
                )
            )
            previous = target
        return float(np.mean(values)) if values else 0.0

    def run(self) -> dict[str, Any]:
        dates = self.store.decision_dates(self.config.warmup_sessions)
        adaptive, _decisions, fixed_params = self._adaptive_schedule(dates)
        fixed = self._fixed_schedule(dates, fixed_params)
        baseline_weights = {
            holding.symbol: holding.weight for holding in self.store.initial_portfolio.holdings
        }
        baseline = {day: baseline_weights for day in dates}
        curves = pd.concat(
            [
                self._simulate("original_baseline", baseline),
                self._simulate("development_selected_fixed", fixed),
                self._simulate("adaptive_agent", adaptive),
            ],
            ignore_index=True,
        )
        curves.to_parquet(self.output / "equity_curves.parquet", index=False)
        curves.to_parquet(self.output / "daily_nav.parquet", index=False)
        projection_records = [
            json.loads(line)
            for line in (self.output / "projections.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        outcome_count = 0
        for projection in projection_records:
            outcome = self._realized_outcome(projection)
            if outcome is not None:
                self.writer.append("forecast_outcomes.jsonl", outcome)
                outcome_count += 1
        metrics = {
            name: performance_metrics(frame.set_index("date")["nav"])
            for name, frame in curves.groupby("strategy")
        }
        split_index = int(len(dates) * self.config.development_fraction)
        holdout_start = dates[split_index]
        evaluation_start = dates[0]
        partition_metrics: dict[str, dict[str, dict[str, float]]] = {
            "development": {},
            "holdout": {},
        }
        for name, frame in curves.groupby("strategy"):
            development = frame[
                (frame["date"] >= evaluation_start) & (frame["date"] < holdout_start)
            ]
            holdout = frame[frame["date"] >= holdout_start]
            partition_metrics["development"][name] = performance_metrics(
                development.set_index("date")["nav"]
            )
            partition_metrics["holdout"][name] = performance_metrics(
                holdout.set_index("date")["nav"]
            )
        average_turnover = {
            "original_baseline": self._average_turnover(baseline),
            "development_selected_fixed": self._average_turnover(fixed),
            "adaptive_agent": self._average_turnover(adaptive),
        }
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.output.parent.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            git_dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.output.parent.parent,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            git_commit = "uncommitted"
            git_dirty = True
        trace_count = sum(
            1
            for line in (self.output / "traces.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
        routing = {
            stage: route.model_dump(mode="json")
            for stage, route in self.config.agent.stage_overrides.items()
        }
        summary = {
            "run_id": self.run_id,
            "dataset_version": self.store.manifest.dataset_version,
            "dataset_hash": self.store.dataset_hash,
            "mode": self.config.agent.mode,
            "provider": self.provider.provider,
            "model": self.provider.model,
            "decision_count": len(dates),
            "development_decisions": int(len(dates) * self.config.development_fraction),
            "holdout_decisions": len(dates) - int(len(dates) * self.config.development_fraction),
            "fixed_strategy_version": fixed_params.version,
            "fixed_strategy_params": fixed_params.__dict__,
            "matured_projection_outcomes": outcome_count,
            "model_call_count": trace_count,
            "routing": routing,
            "metrics": metrics,
            "partition_metrics": partition_metrics,
            "average_monthly_turnover": average_turnover,
            "warning": "Synthetic/retrospective research simulation; not investment advice.",
        }
        self.writer.write("summary.json", summary)
        report = ["# Portfolio Agent Run", "", f"Run: `{self.run_id}`", ""]
        for name, values in metrics.items():
            report.extend([f"## {name}", ""])
            report.extend(f"- {key}: {value:.6f}" for key, value in values.items())
            report.append("")
        report.extend(["## Development and holdout", ""])
        for partition, strategies in partition_metrics.items():
            report.append(f"### {partition.title()}")
            report.append("")
            for name, values in strategies.items():
                report.append(
                    f"- {name}: return {values['total_return']:.6f}; "
                    f"Sharpe {values['sharpe']:.6f}"
                )
            report.append("")
        report.append("Synthetic and retrospective results do not establish investment alpha.\n")
        (self.output / "report.md").write_text("\n".join(report), encoding="utf-8")
        SQLiteLedger(self.output / "run.sqlite3").capture(self.output, summary)
        artifact_hashes = {
            path.name: sha256_path(path)
            for path in sorted(self.output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "run_id": self.run_id,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "dataset_id": self.store.manifest.dataset_version,
            "dataset_hash": self.store.dataset_hash,
            "dataset_files": self.store.manifest.files,
            "seed": self.config.seed,
            "provider": self.provider.provider,
            "model": self.provider.model,
            "routing": routing,
            "generation_settings": self.config.agent.generation.model_dump(mode="json"),
            "budgets": self.config.agent.budgets.model_dump(mode="json"),
            "policy_hash": content_hash(self.config.model_dump(mode="json")),
            "schema_hash": content_hash(self.store.manifest.schema_version),
            "model_call_count": trace_count,
            "fee_bps": self.config.fee_bps,
            "python_version": platform.python_version(),
            "library_versions": {
                package: importlib.metadata.version(package)
                for package in ("pandas", "numpy", "pydantic", "cvxpy")
            },
            "artifact_hashes": artifact_hashes,
        }
        self.writer.write("manifest.json", manifest)
        return summary

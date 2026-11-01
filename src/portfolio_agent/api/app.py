from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from portfolio_agent.agents import build_provider
from portfolio_agent.config import AgentConfig, ReplayConfig
from portfolio_agent.data import DatasetStore
from portfolio_agent.evaluation.outer_replay import ReplayEngine

RUNS_ROOT = Path(os.environ.get("PORTFOLIO_AGENT_RUNS", "runs")).resolve()
DEFAULT_DATASET = Path(
    os.environ.get("PORTFOLIO_AGENT_DATASET", "examples/smoke")
).resolve()
RUN_STATUSES: dict[str, dict] = {}

app = FastAPI(title="Portfolio Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _run_path(run_id: str) -> Path:
    if not RUNS_ROOT.exists():
        raise HTTPException(status_code=404, detail="run not found")
    candidates = sorted(RUNS_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        summary_path = candidate / "summary.json"
        if not candidate.is_dir() or not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_id") == run_id:
            return candidate.resolve()
    raise HTTPException(status_code=404, detail="run not found")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    candidates = [DEFAULT_DATASET, Path("examples/smoke").resolve()]
    values = []
    for path in dict.fromkeys(candidates):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values.append(
            {
                "path": str(path),
                "dataset_version": manifest["dataset_version"],
                "profile": manifest["profile"],
                "is_synthetic": manifest["is_synthetic"],
                "coverage": manifest["coverage"],
            }
        )
    return values


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = str(DEFAULT_DATASET)
    mode: str = "mock"
    provider: str | None = None
    model: str | None = None
    iterations: int = Field(default=3, ge=1, le=5)


def _execute_run(job_id: str, request: RunRequest, output: Path) -> None:
    RUN_STATUSES[job_id]["status"] = "running"
    try:
        agent = AgentConfig(
            mode=request.mode,
            provider=request.provider,
            model=request.model,
        )
        config = ReplayConfig(iterations=request.iterations, agent=agent)
        store = DatasetStore.load(Path(request.dataset).resolve())
        provider = build_provider(config.agent)
        summary = ReplayEngine(store, provider, config, output).run()
        RUN_STATUSES[job_id].update(status="complete", run_id=summary["run_id"])
    except Exception as exc:  # Keep the local queue observable without leaking secrets.
        RUN_STATUSES[job_id].update(status="failed", error=type(exc).__name__)


@app.post("/api/runs", status_code=202)
def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> dict:
    if request.mode not in {"mock", "live"}:
        raise HTTPException(status_code=400, detail="dashboard starts mock or live runs only")
    job_id = uuid.uuid4().hex
    output = RUNS_ROOT / f"job-{job_id[:12]}"
    RUN_STATUSES[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(UTC).isoformat(),
        "output": str(output),
    }
    background_tasks.add_task(_execute_run, job_id, request, output)
    return RUN_STATUSES[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    status = RUN_STATUSES.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@app.get("/api/runs")
def list_runs() -> list[dict]:
    if not RUNS_ROOT.exists():
        return []
    values: dict[str, dict] = {}
    candidates = sorted(RUNS_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if not path.is_dir() or not (path / "summary.json").exists():
            continue
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        values.setdefault(summary["run_id"], summary)
    return list(values.values())


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return json.loads((_run_path(run_id) / "summary.json").read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/months")
def get_months(run_id: str) -> list[dict]:
    return _read_jsonl(_run_path(run_id) / "monthly_decisions.jsonl")


@app.get("/api/runs/{run_id}/months/{month_id}")
def get_month(run_id: str, month_id: str) -> dict:
    root = _run_path(run_id)
    decisions = _read_jsonl(root / "monthly_decisions.jsonl")
    decision = next((item for item in decisions if item["month_id"] == month_id), None)
    if decision is None:
        raise HTTPException(status_code=404, detail="month not found")
    projections = [
        item for item in _read_jsonl(root / "projections.jsonl") if item.get("month_id") == month_id
    ]
    projection_ids = {item["projection_id"] for item in projections}
    return {
        "decision": decision,
        "experiments": [
            item
            for item in _read_jsonl(root / "experiments.jsonl")
            if item.get("month_id") == month_id
        ],
        "projections": projections,
        "outcomes": [
            item
            for item in _read_jsonl(root / "forecast_outcomes.jsonl")
            if item.get("projection_id") in projection_ids
        ],
        "forecasts": [
            item
            for item in _read_jsonl(root / "stock_forecasts.jsonl")
            if item.get("decision_as_of", "").startswith(month_id)
        ],
        "analyses": [
            item
            for item in _read_jsonl(root / "company_analysis.jsonl")
            if item.get("decision_as_of", "").startswith(month_id)
        ],
        "traces": [
            item
            for item in _read_jsonl(root / "traces.jsonl")
            if item.get("decision_as_of", "").startswith(month_id)
        ],
    }


@app.get("/api/runs/{run_id}/experiments")
def get_experiments(run_id: str) -> list[dict]:
    return _read_jsonl(_run_path(run_id) / "experiments.jsonl")


@app.get("/api/runs/{run_id}/equity-curves")
def get_equity_curves(run_id: str) -> list[dict]:
    frame = pd.read_parquet(_run_path(run_id) / "equity_curves.parquet")
    frame["date"] = frame["date"].astype(str)
    return frame.to_dict(orient="records")


@app.get("/api/runs/{run_id}/projections/{projection_id}")
def get_projection(run_id: str, projection_id: str) -> dict:
    items = _read_jsonl(_run_path(run_id) / "projections.jsonl")
    item = next((value for value in items if value["projection_id"] == projection_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="projection not found")
    return item


@app.get("/api/runs/{run_id}/report")
def get_report(run_id: str) -> dict[str, str]:
    return {"markdown": (_run_path(run_id) / "report.md").read_text(encoding="utf-8")}

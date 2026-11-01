from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from portfolio_agent.agents import build_provider
from portfolio_agent.config import AgentConfig, load_config
from portfolio_agent.data import DatasetStore
from portfolio_agent.data.hf_download import download_dataset
from portfolio_agent.evaluation.outer_replay import ReplayEngine
from portfolio_agent.reporting import verify_run

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download(
    repo_id: Annotated[str, typer.Option()],
    revision: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    typer.echo(download_dataset(repo_id, revision, output))


@data_app.command("validate")
def data_validate(dataset: Annotated[Path, typer.Option()]) -> None:
    store = DatasetStore.load(dataset)
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "dataset_version": store.manifest.dataset_version,
                "coverage": store.manifest.coverage,
                "dataset_hash": store.dataset_hash,
                "dataset_repo_id": store.source.repo_id if store.source else None,
                "dataset_revision": store.source.revision if store.source else None,
            },
            indent=2,
        )
    )


def _run(
    dataset: Path,
    output: Path,
    config_path: Path | None,
    mode: str,
    provider: str | None,
    model: str | None,
    iterations: int,
    recording: Path | None,
) -> None:
    config = load_config(config_path)
    agent_data = config.agent.model_dump()
    agent_data.update({"mode": mode})
    if provider:
        agent_data["provider"] = provider
    if model:
        agent_data["model"] = model
    config = config.model_copy(
        update={"agent": AgentConfig.model_validate(agent_data), "iterations": iterations}
    )
    store = DatasetStore.load(dataset)
    provider_client = build_provider(config.agent, recording=recording)
    summary = ReplayEngine(store, provider_client, config, output).run()
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def replay(
    dataset: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    mode: Annotated[str, typer.Option()] = "mock",
    provider: Annotated[str | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    config: Annotated[Path | None, typer.Option()] = None,
    iterations: Annotated[int, typer.Option(min=1, max=5)] = 3,
    recording: Annotated[Path | None, typer.Option()] = None,
    adaptive: Annotated[bool, typer.Option()] = True,
    rebalance: Annotated[str, typer.Option()] = "monthly",
) -> None:
    if not adaptive or rebalance != "monthly":
        raise typer.BadParameter("this release requires adaptive monthly replay")
    _run(dataset, out, config, mode, provider, model, iterations, recording)


@app.command()
def demo(
    dataset: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()] = Path("runs/smoke"),
    mode: Annotated[str, typer.Option()] = "mock",
    iterations: Annotated[int, typer.Option(min=1, max=5)] = 3,
    rebalance: Annotated[str, typer.Option()] = "monthly",
) -> None:
    _run(dataset, out, None, mode, None, None, iterations, None)


@app.command()
def verify(run: Path) -> None:
    result = verify_run(run if run.is_dir() else run.parent)
    typer.echo(json.dumps(result, indent=2))
    if not result["valid"]:
        raise typer.Exit(1)


@app.command()
def report(run: Path) -> None:
    root = run if run.is_dir() else run.parent
    typer.echo((root / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()

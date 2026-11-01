from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from portfolio_agent.evaluation.analytics import performance_metrics
from portfolio_agent.persistence.artifact_writer import sha256_path


def verify_run(run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    curves = pd.read_parquet(run_dir / "equity_curves.parquet")
    recomputed = {
        name: performance_metrics(frame.set_index("date")["nav"])
        for name, frame in curves.groupby("strategy")
    }
    errors = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_hashes = manifest.get("artifact_hashes", {})
        if not artifact_hashes:
            errors.append("manifest contains no artifact hashes")
        for name, expected_hash in artifact_hashes.items():
            path = run_dir / name
            if not path.exists():
                errors.append(f"missing artifact: {name}")
            elif sha256_path(path) != expected_hash:
                errors.append(f"artifact hash mismatch: {name}")
    for strategy, expected in summary["metrics"].items():
        actual = recomputed.get(strategy)
        if actual is None:
            errors.append(f"missing strategy curve: {strategy}")
            continue
        for metric, expected_value in expected.items():
            if abs(actual[metric] - expected_value) > 1e-10:
                errors.append(f"metric mismatch: {strategy}.{metric}")
    return {"valid": not errors, "errors": errors, "recomputed_metrics": recomputed}

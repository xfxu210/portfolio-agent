import json
import re
from pathlib import Path

from huggingface_hub import snapshot_download

SOURCE_METADATA_FILE = ".portfolio-agent-source.json"


def download_dataset(repo_id: str, revision: str, output: Path) -> Path:
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("revision must be a full immutable commit SHA")
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output,
    )
    root = Path(downloaded)
    (root / SOURCE_METADATA_FILE).write_text(
        json.dumps(
            {"repo_id": repo_id, "revision": revision.lower()},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return root

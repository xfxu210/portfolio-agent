from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    def encode(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        return str(item)

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=encode)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, value: Any) -> str:
        encoded = canonical_json(value)
        with (self.root / name).open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        return hashlib.sha256(encoded.encode()).hexdigest()

    def write(self, name: str, value: Any) -> str:
        encoded = canonical_json(value) + "\n"
        (self.root / name).write_text(encoded, encoding="utf-8")
        return hashlib.sha256(encoded.encode()).hexdigest()

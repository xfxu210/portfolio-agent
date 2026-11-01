from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .base import ProviderError, StructuredResult

T = TypeVar("T", bound=BaseModel)


class RecordedProvider:
    provider = "recorded"

    def __init__(self, recording: Path) -> None:
        self.model = "recorded"
        self.records = {
            item["request_id"]: item
            for item in (
                json.loads(line)
                for line in recording.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    def generate_structured(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        request_id: str,
    ) -> StructuredResult[T]:
        del task, messages
        item = self.records.get(request_id)
        if item is None:
            raise ProviderError("recording_missing", f"no recorded response for {request_id}")
        # Validate through JSON so strict tuple/enum fields have the same semantics
        # as a live provider response.
        value = response_model.model_validate_json(json.dumps(item["normalized"]))
        return StructuredResult(
            value=value,
            provider=item["provider"],
            model=item["model"],
            request_id=item.get("provider_request_id"),
            latency_ms=item.get("latency_ms", 0),
            input_tokens=item.get("input_tokens"),
            output_tokens=item.get("output_tokens"),
            raw_hash=item["output_hash"],
        )

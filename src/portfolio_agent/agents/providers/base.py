from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    provider: str
    model: str
    request_id: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    raw_hash: str


class LLMProvider(Protocol):
    provider: str
    model: str

    def generate_structured(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        request_id: str,
    ) -> StructuredResult[T]: ...


class JsonProvider:
    provider: str
    model: str
    max_retries: int = 0
    retry_base_seconds: float = 0.25

    def _invoke(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        request_id: str,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def generate_structured(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        request_id: str,
    ) -> StructuredResult[T]:
        started = time.monotonic()
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw, metadata = self._invoke(
                    task=task,
                    messages=messages,
                    schema=response_model.model_json_schema(),
                    request_id=request_id,
                )
                try:
                    value = response_model.model_validate_json(raw)
                except ValidationError as exc:
                    raise ProviderError(
                        "schema_validation", str(exc), retryable=True
                    ) from exc
                break
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_base_seconds * (2**attempt))
        else:  # pragma: no cover - defensive; loop either succeeds or raises.
            assert last_error is not None
            raise last_error
        latency_ms = int((time.monotonic() - started) * 1000)
        return StructuredResult(
            value=value,
            provider=self.provider,
            model=self.model,
            request_id=metadata.get("request_id"),
            latency_ms=latency_ms,
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            raw_hash=hashlib.sha256(raw.encode()).hexdigest(),
        )


def schema_instruction(task: str, schema: dict[str, Any]) -> str:
    return (
        f"Task: {task}. Return only one JSON object matching this JSON Schema. "
        "Do not include markdown or additional keys. Schema: " + json.dumps(schema, sort_keys=True)
    )

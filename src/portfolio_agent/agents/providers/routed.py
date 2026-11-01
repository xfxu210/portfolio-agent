from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .base import LLMProvider, StructuredResult

T = TypeVar("T", bound=BaseModel)


class RoutedProvider:
    """Dispatches each agent stage to its explicitly configured provider/model."""

    provider = "routed"
    model = "per-stage"

    def __init__(self, default: LLMProvider, routes: dict[str, LLMProvider]) -> None:
        self.default = default
        self.routes = routes

    def generate_structured(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        request_id: str,
    ) -> StructuredResult[T]:
        provider = self.routes.get(task, self.default)
        return provider.generate_structured(
            task=task,
            messages=messages,
            response_model=response_model,
            request_id=request_id,
        )

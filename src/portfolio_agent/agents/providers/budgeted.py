from __future__ import annotations

from collections import defaultdict
from typing import TypeVar

from pydantic import BaseModel

from portfolio_agent.config import BudgetConfig

from .base import LLMProvider, ProviderError, StructuredResult

T = TypeVar("T", bound=BaseModel)


class BudgetedProvider:
    """Enforces deterministic per-month call and run-wide token ceilings."""

    def __init__(self, provider: LLMProvider, budget: BudgetConfig) -> None:
        self.inner = provider
        self.provider = provider.provider
        self.model = provider.model
        self.budget = budget
        self.calls_by_month: dict[str, int] = defaultdict(int)
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def _month(request_id: str) -> str:
        parts = request_id.split(":")
        return parts[1][:7] if len(parts) > 1 else "run"

    def generate_structured(
        self,
        *,
        task: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        request_id: str,
    ) -> StructuredResult[T]:
        month = self._month(request_id)
        if self.calls_by_month[month] >= self.budget.max_model_calls_per_month:
            raise ProviderError("budget_exceeded", f"model call budget exceeded for {month}")
        if self.input_tokens >= self.budget.max_total_input_tokens:
            raise ProviderError("budget_exceeded", "input token budget exceeded")
        if self.output_tokens >= self.budget.max_total_output_tokens:
            raise ProviderError("budget_exceeded", "output token budget exceeded")
        result = self.inner.generate_structured(
            task=task,
            messages=messages,
            response_model=response_model,
            request_id=request_id,
        )
        self.calls_by_month[month] += 1
        self.input_tokens += result.input_tokens or 0
        self.output_tokens += result.output_tokens or 0
        return result

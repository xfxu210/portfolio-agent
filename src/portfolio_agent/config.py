from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationConfig(StrictConfig):
    max_output_tokens: int = Field(default=16_000, ge=1)
    timeout_seconds: int = Field(default=90, ge=1)
    max_retries: int = Field(default=2, ge=0, le=5)


class BudgetConfig(StrictConfig):
    max_model_calls_per_month: int = Field(default=150, ge=1)
    max_total_input_tokens: int = Field(default=5_000_000, ge=1)
    max_total_output_tokens: int = Field(default=500_000, ge=1)


class ReasoningConfig(StrictConfig):
    max_steps: int = Field(default=7, ge=2, le=12)
    min_information_gain: float = Field(default=0.05, ge=0.0, le=1.0)
    max_revision_symbols: int = Field(default=3, ge=1, le=5)


class StageRoute(StrictConfig):
    provider: Literal["openai", "anthropic", "gemini", "muse"]
    model: str = Field(min_length=1)


class AgentConfig(StrictConfig):
    mode: Literal["mock", "live", "recorded"] = "mock"
    provider: Literal["openai", "anthropic", "gemini", "muse"] | None = None
    model: str | None = None
    stage_overrides: dict[str, StageRoute] = {}
    generation: GenerationConfig = GenerationConfig()
    budgets: BudgetConfig = BudgetConfig()
    reasoning: ReasoningConfig = ReasoningConfig()

    @model_validator(mode="after")
    def live_requires_explicit_model(self) -> AgentConfig:
        if self.mode == "live" and (not self.provider or not self.model):
            raise ValueError("live mode requires an explicit provider and model")
        return self


class ReplayConfig(StrictConfig):
    seed: int = 42
    iterations: int = Field(default=3, ge=1, le=5)
    fee_bps: Literal[0.0] = 0.0
    warmup_sessions: int = Field(default=252, ge=126)
    development_fraction: float = Field(default=0.6, gt=0.2, lt=0.9)
    agent: AgentConfig = AgentConfig()


def load_config(path: Path | None) -> ReplayConfig:
    if path is None:
        return ReplayConfig()
    return ReplayConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

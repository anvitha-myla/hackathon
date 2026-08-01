"""AI explainer output schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class OperatorAdvisory(BaseModel):
    """Concise operator note from LocalAIExplainer."""

    model_config = ConfigDict(extra="forbid")

    advisory_text: str = Field(..., description="≤3 sentence operator note")
    source: Literal["ollama", "template_fallback"] = Field(
        ...,
        description="Whether text came from local LLM or offline template",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Ollama model tag used, if any",
    )
    stage: Optional[str] = Field(default=None)
    ekf_confidence: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    safety_status: Optional[str] = Field(default=None)
    timed_out: bool = Field(False)
    ollama_reachable: bool = Field(False)
    metadata: dict[str, Any] = Field(default_factory=dict)

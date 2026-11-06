"""
schemas.py — Pydantic models for Flora Magic agent outputs.

Both agents must return responses that satisfy these schemas before
their output is passed to the next stage of the workflow.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pathologist Agent output schema
# ---------------------------------------------------------------------------

class DiagnosisReport(BaseModel):
    """Structured output from the Pathologist Agent (vision model)."""

    agent: Literal["Pathologist Agent"]
    image_quality: Literal["acceptable", "poor", "unacceptable"]
    crop_name: str = Field(min_length=1)
    suspected_condition: str = Field(min_length=1)
    condition_category: str = Field(min_length=1)
    observed_symptoms: List[str] = Field(min_length=1)
    possible_causes: List[str] = Field(min_length=1)
    alternative_diagnoses: List[str] = Field(default_factory=list)
    confidence_level: Literal[
        "strong visual indications",
        "moderate visual indications",
        "insufficient evidence",
    ]
    requires_expert_review: bool
    status: Literal[
        "preliminary_assessment",
        "insufficient_image_quality",
        "unclear",
    ]

    @field_validator("requires_expert_review")
    @classmethod
    def must_require_expert_review(cls, v: bool) -> bool:
        """Safety guardrail: expert review must always be flagged."""
        if not v:
            raise ValueError(
                "requires_expert_review must be true — Flora Magic never replaces an expert."
            )
        return v

    @field_validator("observed_symptoms", "possible_causes", mode="before")
    @classmethod
    def non_empty_list(cls, v: list) -> list:
        if not v:
            raise ValueError("List must contain at least one item.")
        return v


# ---------------------------------------------------------------------------
# Extension Officer Agent output schema
# ---------------------------------------------------------------------------

class AdvisoryReport(BaseModel):
    """Structured output from the Extension Officer Agent (text model)."""

    agent: Literal["Extension Officer Agent"]
    crop_health_summary: str = Field(min_length=10)
    immediate_actions: List[str] = Field(min_length=1)
    preventive_measures: List[str] = Field(min_length=1)
    crop_care_recommendations: List[str] = Field(min_length=1)
    treatment_guidance: str = Field(min_length=10)
    expert_consultation: str = Field(min_length=10)
    safety_warning: str = Field(min_length=10)

    @field_validator(
        "immediate_actions", "preventive_measures", "crop_care_recommendations",
        mode="before",
    )
    @classmethod
    def non_empty_list(cls, v: list) -> list:
        if not v:
            raise ValueError("List must contain at least one item.")
        return v


# ---------------------------------------------------------------------------
# Execution metrics (session-level, not persisted)
# ---------------------------------------------------------------------------

class AgentExecution(BaseModel):
    """Records one agent's execution event for the performance dashboard."""

    agent: str
    status: Literal["completed", "failed", "skipped"]
    latency_seconds: float = 0.0
    validation: Literal["passed", "failed", "skipped"] = "skipped"
    input_tokens: int = 0
    output_tokens: int = 0
    guardrail_interventions: int = 0
    error_message: Optional[str] = None


class SessionMetrics(BaseModel):
    """Aggregated session-level performance metrics."""

    total_requests: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0
    total_guardrail_interventions: int = 0
    total_api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    executions: List[AgentExecution] = Field(default_factory=list)

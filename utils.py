"""
utils.py — Image preparation, report formatting, session metrics helpers,
and general utilities for Flora Magic.
"""

import datetime
import io
import logging
from typing import Optional

from PIL import Image

from schemas import AgentExecution, SessionMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image_bytes(uploaded_file) -> bytes:
    """Read bytes from a Streamlit UploadedFile object."""
    uploaded_file.seek(0)
    return uploaded_file.read()


def image_to_display(image_bytes: bytes):
    """Return a PIL Image suitable for st.image()."""
    return Image.open(io.BytesIO(image_bytes))


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def build_text_report(
    diagnosis: dict,
    advisory: dict,
    crop_name: str = "",
    location: str = "",
) -> str:
    """
    Assemble the final downloadable advisory report as plain text.
    Does NOT include API credentials, prompts, or internal data.
    """
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines.append("=" * 60)
    lines.append("       🌿 FLORA MAGIC — CROP HEALTH ADVISORY REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {now}")
    lines.append("")

    # --- Diagnosis section ---
    lines.append("SECTION 1 — DIAGNOSTIC FINDINGS")
    lines.append("-" * 40)
    lines.append(f"Crop Name        : {diagnosis.get('crop_name', crop_name or 'Not specified')}")
    lines.append(f"Location         : {location or 'Not specified'}")
    lines.append(f"Suspected Condition : {diagnosis.get('suspected_condition', 'N/A')}")
    lines.append(f"Condition Category  : {diagnosis.get('condition_category', 'N/A')}")
    lines.append(f"Image Quality    : {diagnosis.get('image_quality', 'N/A')}")
    lines.append(f"Confidence Level : {diagnosis.get('confidence_level', 'N/A')}")
    lines.append(f"Expert Review Required: {'Yes' if diagnosis.get('requires_expert_review') else 'No'}")
    lines.append("")

    symptoms = diagnosis.get("observed_symptoms", [])
    if symptoms:
        lines.append("Observed Symptoms:")
        for s in symptoms:
            lines.append(f"  • {s}")
    lines.append("")

    causes = diagnosis.get("possible_causes", [])
    if causes:
        lines.append("Possible Causes:")
        for c in causes:
            lines.append(f"  • {c}")
    lines.append("")

    alternatives = diagnosis.get("alternative_diagnoses", [])
    if alternatives:
        lines.append("Alternative Diagnoses:")
        for a in alternatives:
            lines.append(f"  • {a}")
    lines.append("")

    # --- Advisory section ---
    lines.append("SECTION 2 — AGRICULTURAL ADVISORY")
    lines.append("-" * 40)
    lines.append("Crop Health Summary:")
    lines.append(advisory.get("crop_health_summary", "N/A"))
    lines.append("")

    immediate = advisory.get("immediate_actions", [])
    if immediate:
        lines.append("Immediate Actions:")
        for a in immediate:
            lines.append(f"  • {a}")
    lines.append("")

    preventive = advisory.get("preventive_measures", [])
    if preventive:
        lines.append("Preventive Measures:")
        for p in preventive:
            lines.append(f"  • {p}")
    lines.append("")

    care = advisory.get("crop_care_recommendations", [])
    if care:
        lines.append("Crop-Care Recommendations:")
        for r in care:
            lines.append(f"  • {r}")
    lines.append("")

    lines.append("Treatment Guidance:")
    lines.append(advisory.get("treatment_guidance", "Consult a local agricultural expert."))
    lines.append("")

    lines.append("When to Consult an Expert:")
    lines.append(advisory.get("expert_consultation", "Contact your local agricultural extension officer."))
    lines.append("")

    lines.append("Safety Warning:")
    lines.append(advisory.get("safety_warning", "Follow all local agricultural regulations."))
    lines.append("")

    # --- Disclaimer ---
    lines.append("=" * 60)
    lines.append("DISCLAIMER")
    lines.append("-" * 40)
    lines.append(
        "This report is generated using AI and represents a preliminary visual assessment.\n"
        "It is NOT a confirmed laboratory diagnosis.\n"
        "Consult a qualified agricultural professional before making significant treatment decisions.\n"
        "Flora Magic is a decision-support tool and does not replace expert agronomic advice."
    )
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session metrics helpers
# ---------------------------------------------------------------------------

def init_session_metrics() -> SessionMetrics:
    """Return a fresh SessionMetrics object."""
    return SessionMetrics()


def record_workflow_result(
    metrics: SessionMetrics,
    pathologist_exec: Optional[AgentExecution],
    extension_exec: Optional[AgentExecution],
    success: bool,
) -> SessionMetrics:
    """
    Update the SessionMetrics after a workflow run.
    Returns the updated metrics (mutated in-place for simplicity).
    """
    metrics.total_requests += 1
    if success:
        metrics.successful_workflows += 1
    else:
        metrics.failed_workflows += 1

    for exec_record in filter(None, [pathologist_exec, extension_exec]):
        metrics.executions.append(exec_record)
        metrics.total_api_calls += 1
        metrics.total_input_tokens  += exec_record.input_tokens
        metrics.total_output_tokens += exec_record.output_tokens
        metrics.total_guardrail_interventions += exec_record.guardrail_interventions

    return metrics


def average_latency(metrics: SessionMetrics) -> float:
    """Return average total workflow latency across completed executions."""
    completed = [
        e.latency_seconds
        for e in metrics.executions
        if e.status == "completed"
    ]
    return sum(completed) / len(completed) if completed else 0.0


def execution_log_rows(metrics: SessionMetrics) -> list[dict]:
    """Return a list of dicts suitable for st.dataframe()."""
    rows = []
    for e in metrics.executions:
        rows.append({
            "Agent": e.agent,
            "Status": e.status.capitalize(),
            "Execution Time (s)": f"{e.latency_seconds:.2f}",
            "Validation": e.validation.capitalize(),
            "Input Tokens": e.input_tokens or "—",
            "Output Tokens": e.output_tokens or "—",
            "Guardrail Interventions": e.guardrail_interventions,
        })
    return rows

"""
agents.py — Pathologist Agent and Extension Officer Agent for Flora Magic.

Each agent is a self-contained class that:
 1. Calls the appropriate IBM model via ibm_client.py
 2. Applies guardrails from guardrails.py
 3. Returns a validated structured result

The two agents run sequentially:
    PathologistAgent.analyze()  →  ExtensionOfficerAgent.generate()
"""

import logging
import time
from typing import Optional

from ibm_watsonx_ai import APIClient

import ibm_client
import guardrails as gr
from schemas import AgentExecution

logger = logging.getLogger(__name__)

# Maximum number of inference retries per agent
MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# Pathologist Agent
# ---------------------------------------------------------------------------

class PathologistAgent:
    """
    Invokes an IBM Granite Vision model to analyse a crop image and
    return a structured diagnostic JSON.

    This agent may ONLY:
      - Process validated image bytes
      - Call the configured IBM vision model
      - Return a structured DiagnosisReport dict

    It may NOT access files, secrets, or external APIs beyond the model call.
    """

    def __init__(
        self,
        client: APIClient,
        project_id: str,
        vision_model_id: str,
        max_new_tokens: int = 1024,
    ):
        self.client = client
        self.project_id = project_id
        self.vision_model_id = vision_model_id
        self.max_new_tokens = max_new_tokens

    def analyze(
        self,
        image_bytes: bytes,
        crop_name: str = "",
        symptoms: str = "",
        location: str = "",
    ) -> dict:
        """
        Run crop image analysis.

        Returns a dict with keys:
            diagnosis     — validated DiagnosisReport dict (or None on failure)
            execution     — AgentExecution metrics
            error         — error message string (or None)
        """
        execution = AgentExecution(agent="Pathologist Agent", status="failed")
        t_start = time.monotonic()

        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info(
                    "Pathologist Agent: calling vision model '%s' (attempt %d).",
                    self.vision_model_id, attempt + 1,
                )

                response = ibm_client.analyze_crop_image(
                    client=self.client,
                    project_id=self.project_id,
                    vision_model_id=self.vision_model_id,
                    image_bytes=image_bytes,
                    crop_name=crop_name,
                    symptoms=symptoms,
                    location=location,
                    max_new_tokens=self.max_new_tokens,
                )

                raw_text = response.get("raw_text", "")
                usage = response.get("usage", {})
                model_latency = response.get("latency", 0.0)

                # Update token counts
                execution.input_tokens  = usage.get("prompt_tokens", 0)
                execution.output_tokens = usage.get("completion_tokens", 0)

                # Validate output
                result = gr.validate_diagnosis_output(raw_text)

                if not result.ok:
                    logger.warning(
                        "Pathologist output validation failed (attempt %d): %s",
                        attempt + 1, result.message,
                    )
                    if result.intervention:
                        execution.guardrail_interventions += 1
                    if attempt < MAX_RETRIES:
                        logger.info("Retrying Pathologist Agent...")
                        continue
                    # All retries exhausted
                    execution.status = "failed"
                    execution.validation = "failed"
                    execution.latency_seconds = time.monotonic() - t_start
                    return {
                        "diagnosis": None,
                        "execution": execution,
                        "error": result.message,
                    }

                # Success
                execution.status = "completed"
                execution.validation = "passed"
                execution.latency_seconds = time.monotonic() - t_start
                logger.info("Pathologist Agent completed successfully.")
                return {
                    "diagnosis": result.data,
                    "execution": execution,
                    "error": None,
                }

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Pathologist Agent error (attempt %d): %s",
                    attempt + 1, exc,
                )
                if attempt < MAX_RETRIES:
                    logger.info("Retrying Pathologist Agent after error...")
                    time.sleep(1)
                    continue
                execution.status = "failed"
                execution.latency_seconds = time.monotonic() - t_start
                execution.error_message = _safe_error_message(exc)
                return {
                    "diagnosis": None,
                    "execution": execution,
                    "error": execution.error_message,
                }

        # Should not reach here
        execution.latency_seconds = time.monotonic() - t_start
        return {"diagnosis": None, "execution": execution, "error": "Unknown failure."}


# ---------------------------------------------------------------------------
# Extension Officer Agent
# ---------------------------------------------------------------------------

class ExtensionOfficerAgent:
    """
    Invokes IBM Granite 4 H Small to convert a validated diagnosis into a
    practical, farmer-friendly advisory.

    This agent may ONLY:
      - Read the validated diagnosis dict passed to it
      - Call the configured IBM text model
      - Return a structured AdvisoryReport dict

    It may NOT invent a diagnosis, access files, secrets, or external APIs.
    """

    def __init__(
        self,
        client: APIClient,
        project_id: str,
        advisory_model_id: str,
        max_new_tokens: int = 1024,
    ):
        self.client = client
        self.project_id = project_id
        self.advisory_model_id = advisory_model_id
        self.max_new_tokens = max_new_tokens

    def generate(self, diagnosis: dict) -> dict:
        """
        Generate a farmer advisory from a validated diagnosis.

        Returns a dict with keys:
            advisory   — validated AdvisoryReport dict (or None on failure)
            execution  — AgentExecution metrics
            error      — error message string (or None)
        """
        execution = AgentExecution(agent="Extension Officer Agent", status="failed")
        t_start = time.monotonic()

        # Multi-agent handoff guardrail — must pass before calling the model
        handoff_check = gr.validate_agent_handoff(diagnosis)
        if not handoff_check.ok:
            execution.status = "skipped"
            execution.validation = "failed"
            execution.guardrail_interventions += 1
            execution.latency_seconds = time.monotonic() - t_start
            return {
                "advisory": None,
                "execution": execution,
                "error": handoff_check.message,
            }

        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info(
                    "Extension Officer Agent: calling advisory model '%s' (attempt %d).",
                    self.advisory_model_id, attempt + 1,
                )

                response = ibm_client.generate_farmer_advisory(
                    client=self.client,
                    project_id=self.project_id,
                    advisory_model_id=self.advisory_model_id,
                    diagnosis=diagnosis,
                    max_new_tokens=self.max_new_tokens,
                )

                raw_text = response.get("raw_text", "")
                usage = response.get("usage", {})

                execution.input_tokens  = usage.get("prompt_tokens", 0)
                execution.output_tokens = usage.get("completion_tokens", 0)

                # Validate output (includes unsafe chemical check)
                result = gr.validate_advisory_output(raw_text, diagnosis)

                if not result.ok:
                    logger.warning(
                        "Extension Officer output validation failed (attempt %d): %s",
                        attempt + 1, result.message,
                    )
                    if result.intervention:
                        execution.guardrail_interventions += 1
                    if attempt < MAX_RETRIES:
                        logger.info("Retrying Extension Officer Agent...")
                        continue
                    execution.status = "failed"
                    execution.validation = "failed"
                    execution.latency_seconds = time.monotonic() - t_start
                    return {
                        "advisory": None,
                        "execution": execution,
                        "error": result.message,
                    }

                # Success
                execution.status = "completed"
                execution.validation = "passed"
                execution.latency_seconds = time.monotonic() - t_start
                logger.info("Extension Officer Agent completed successfully.")
                return {
                    "advisory": result.data,
                    "execution": execution,
                    "error": None,
                }

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Extension Officer Agent error (attempt %d): %s",
                    attempt + 1, exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue
                execution.status = "failed"
                execution.latency_seconds = time.monotonic() - t_start
                execution.error_message = _safe_error_message(exc)
                return {
                    "advisory": None,
                    "execution": execution,
                    "error": execution.error_message,
                }

        execution.latency_seconds = time.monotonic() - t_start
        return {"advisory": None, "execution": execution, "error": "Unknown failure."}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_error_message(exc: Exception) -> str:
    """
    Return a user-friendly error string while hiding credentials or
    internal details that should not surface in the UI.
    """
    msg = str(exc)
    # Redact anything that looks like a token or key
    import re
    msg = re.sub(r"(api[_-]?key|apikey|token|bearer)\s*[=:]\s*\S+", "[REDACTED]", msg, flags=re.IGNORECASE)
    # IBM API errors often include useful status info — keep the first sentence
    lines = msg.splitlines()
    first_line = lines[0] if lines else "An unexpected error occurred."
    # Generic messages for known error types
    if "401" in msg or "Unauthorized" in msg:
        return "IBM API authentication failed. Please check your API key and project ID."
    if "403" in msg or "Forbidden" in msg:
        return "Access denied. Verify that your project ID and API key have the required permissions."
    if "404" in msg or "not found" in msg.lower():
        return "The configured model ID was not found. Please verify the model name in the sidebar."
    if "429" in msg or "rate limit" in msg.lower():
        return "IBM API rate limit reached. Please wait a moment and try again."
    if "timeout" in msg.lower() or "timed out" in msg.lower():
        return "Request timed out. Check your network connection and try again."
    if "quota" in msg.lower():
        return "IBM API quota exceeded. Please check your account usage limits."
    return f"API error: {first_line}"

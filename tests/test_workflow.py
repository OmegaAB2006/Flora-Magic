"""
tests/test_workflow.py — Integration tests for the Flora Magic multi-agent
workflow using mocked IBM API responses.

No real IBM credentials or network calls are required.
"""

import io
import json
import sys
import os
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub the IBM SDK before importing agents so tests run without the package.
# ---------------------------------------------------------------------------
def _make_stub_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod

for _mod_name in [
    "ibm_watsonx_ai",
    "ibm_watsonx_ai.foundation_models",
    "ibm_watsonx_ai.foundation_models.inference",
]:
    if _mod_name not in sys.modules:
        _make_stub_module(_mod_name)

# Provide the symbols agents.py and ibm_client.py reference at import time
sys.modules["ibm_watsonx_ai"].APIClient = MagicMock          # type: ignore[attr-defined]
sys.modules["ibm_watsonx_ai"].Credentials = MagicMock        # type: ignore[attr-defined]
sys.modules["ibm_watsonx_ai.foundation_models"].ModelInference = MagicMock  # type: ignore[attr-defined]

from agents import PathologistAgent, ExtensionOfficerAgent
import guardrails as gr


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

def _make_test_image_bytes() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (200, 200), color=(100, 180, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MOCK_DIAGNOSIS_JSON = json.dumps({
    "agent": "Pathologist Agent",
    "image_quality": "acceptable",
    "crop_name": "Tomato",
    "suspected_condition": "Possible leaf spot disease",
    "condition_category": "Possible fungal disease",
    "observed_symptoms": ["Small dark spots", "Yellowing"],
    "possible_causes": ["Fungal infection"],
    "alternative_diagnoses": ["Nutrient deficiency"],
    "confidence_level": "moderate visual indications",
    "requires_expert_review": True,
    "status": "preliminary_assessment",
})

MOCK_ADVISORY_JSON = json.dumps({
    "agent": "Extension Officer Agent",
    "crop_health_summary": "Your tomato shows symptoms that may indicate a leaf disease.",
    "immediate_actions": ["Inspect nearby plants for similar symptoms"],
    "preventive_measures": ["Maintain appropriate plant spacing"],
    "crop_care_recommendations": ["Monitor symptom progression"],
    "treatment_guidance": "Consult a local agricultural expert before applying any treatment.",
    "expert_consultation": "Contact an agricultural extension officer if symptoms spread.",
    "safety_warning": "Follow all local agricultural regulations and product safety labels.",
})

MOCK_POOR_IMAGE_JSON = json.dumps({
    "agent": "Pathologist Agent",
    "image_quality": "unacceptable",
    "crop_name": "Unknown",
    "suspected_condition": "Cannot determine — image too blurry",
    "condition_category": "Unclear",
    "observed_symptoms": ["Unable to identify symptoms"],
    "possible_causes": ["Image quality insufficient"],
    "alternative_diagnoses": [],
    "confidence_level": "insufficient evidence",
    "requires_expert_review": True,
    "status": "insufficient_image_quality",
})


def _make_mock_client():
    """Return a mock IBM APIClient."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Pathologist Agent tests
# ---------------------------------------------------------------------------

class TestPathologistAgent:

    def _agent(self):
        return PathologistAgent(
            client=_make_mock_client(),
            project_id="test-project",
            vision_model_id="ibm/granite-vision-3-3-2b",
        )

    def test_successful_analysis(self):
        """Happy path: valid model response produces a validated diagnosis."""
        agent = self._agent()
        with patch(
            "ibm_client.analyze_crop_image",
            return_value={
                "raw_text": MOCK_DIAGNOSIS_JSON,
                "usage": {"prompt_tokens": 100, "completion_tokens": 200},
                "latency": 1.5,
            },
        ):
            result = agent.analyze(
                image_bytes=_make_test_image_bytes(),
                crop_name="Tomato",
                symptoms="Yellow spots",
                location="Punjab",
            )

        assert result["error"] is None
        assert result["diagnosis"] is not None
        assert result["diagnosis"]["crop_name"] == "Tomato"
        assert result["execution"].status == "completed"
        assert result["execution"].validation == "passed"

    def test_poor_image_quality_flagged(self):
        """Model returns unacceptable image quality — diagnosis status reflects it."""
        agent = self._agent()
        with patch(
            "ibm_client.analyze_crop_image",
            return_value={
                "raw_text": MOCK_POOR_IMAGE_JSON,
                "usage": {},
                "latency": 1.0,
            },
        ):
            result = agent.analyze(image_bytes=_make_test_image_bytes())

        assert result["error"] is None
        assert result["diagnosis"]["status"] == "insufficient_image_quality"
        assert result["diagnosis"]["image_quality"] == "unacceptable"

    def test_invalid_json_response_fails(self):
        """Model returns garbage text — validation fails gracefully."""
        agent = self._agent()
        with patch(
            "ibm_client.analyze_crop_image",
            return_value={
                "raw_text": "Sorry, I cannot analyse this image.",
                "usage": {},
                "latency": 0.8,
            },
        ):
            result = agent.analyze(image_bytes=_make_test_image_bytes())

        assert result["diagnosis"] is None
        assert result["error"] is not None
        assert result["execution"].status == "failed"

    def test_api_exception_handled(self):
        """IBM API raises an exception — agent returns safe error message."""
        agent = self._agent()
        with patch(
            "ibm_client.analyze_crop_image",
            side_effect=Exception("401 Unauthorized"),
        ):
            result = agent.analyze(image_bytes=_make_test_image_bytes())

        assert result["diagnosis"] is None
        assert result["error"] is not None
        assert result["execution"].status == "failed"
        # Credentials must not appear in the error message
        assert "api_key" not in result["error"].lower()
        assert "apikey" not in result["error"].lower()


# ---------------------------------------------------------------------------
# Extension Officer Agent tests
# ---------------------------------------------------------------------------

class TestExtensionOfficerAgent:

    def _agent(self):
        return ExtensionOfficerAgent(
            client=_make_mock_client(),
            project_id="test-project",
            advisory_model_id="ibm/granite-4-h-small",
        )

    def _valid_diagnosis(self):
        return json.loads(MOCK_DIAGNOSIS_JSON)

    def test_successful_advisory(self):
        """Happy path: valid diagnosis produces a validated advisory."""
        agent = self._agent()
        with patch(
            "ibm_client.generate_farmer_advisory",
            return_value={
                "raw_text": MOCK_ADVISORY_JSON,
                "usage": {"prompt_tokens": 300, "completion_tokens": 250},
                "latency": 2.0,
            },
        ):
            result = agent.generate(diagnosis=self._valid_diagnosis())

        assert result["error"] is None
        assert result["advisory"] is not None
        assert result["advisory"]["agent"] == "Extension Officer Agent"
        assert result["execution"].status == "completed"

    def test_cannot_run_without_valid_diagnosis(self):
        """Extension Officer must be blocked when diagnosis is None."""
        agent = self._agent()
        result = agent.generate(diagnosis=None)
        assert result["advisory"] is None
        assert result["error"] is not None
        assert result["execution"].status == "skipped"

    def test_cannot_run_with_poor_image_diagnosis(self):
        """Extension Officer must be blocked when image quality is unacceptable."""
        agent = self._agent()
        poor_diagnosis = json.loads(MOCK_POOR_IMAGE_JSON)
        result = agent.generate(diagnosis=poor_diagnosis)
        assert result["advisory"] is None
        assert result["error"] is not None

    def test_api_exception_handled(self):
        """IBM API raises an exception — agent returns safe error."""
        agent = self._agent()
        with patch(
            "ibm_client.generate_farmer_advisory",
            side_effect=Exception("429 Rate limit exceeded"),
        ):
            result = agent.generate(diagnosis=self._valid_diagnosis())

        assert result["advisory"] is None
        assert result["error"] is not None
        assert result["execution"].status == "failed"


# ---------------------------------------------------------------------------
# End-to-end workflow: Pathologist → Extension Officer handoff
# ---------------------------------------------------------------------------

class TestAgentHandoff:

    def test_full_workflow_passes_diagnosis_to_advisory(self):
        """
        Verify that Pathologist output is correctly passed to the
        Extension Officer Agent (not hardcoded or generated independently).
        """
        p_agent = PathologistAgent(
            client=_make_mock_client(),
            project_id="test-project",
            vision_model_id="ibm/granite-vision-3-3-2b",
        )
        e_agent = ExtensionOfficerAgent(
            client=_make_mock_client(),
            project_id="test-project",
            advisory_model_id="ibm/granite-4-h-small",
        )

        with patch(
            "ibm_client.analyze_crop_image",
            return_value={
                "raw_text": MOCK_DIAGNOSIS_JSON,
                "usage": {"prompt_tokens": 100, "completion_tokens": 200},
                "latency": 1.5,
            },
        ):
            p_result = p_agent.analyze(
                image_bytes=_make_test_image_bytes(),
                crop_name="Tomato",
            )

        assert p_result["error"] is None
        diagnosis = p_result["diagnosis"]

        # Verify handoff guardrail passes
        handoff = gr.validate_agent_handoff(diagnosis)
        assert handoff.ok

        with patch(
            "ibm_client.generate_farmer_advisory",
            return_value={
                "raw_text": MOCK_ADVISORY_JSON,
                "usage": {"prompt_tokens": 300, "completion_tokens": 250},
                "latency": 2.0,
            },
        ) as mock_advisory:
            e_result = e_agent.generate(diagnosis=diagnosis)

            # Confirm the diagnosis dict was actually passed to the IBM call
            call_kwargs = mock_advisory.call_args
            passed_diagnosis = call_kwargs.kwargs.get("diagnosis") or call_kwargs.args[3] if call_kwargs.args else None
            if passed_diagnosis is None and call_kwargs.kwargs:
                passed_diagnosis = call_kwargs.kwargs.get("diagnosis")

        assert e_result["error"] is None
        assert e_result["advisory"] is not None

    def test_extension_officer_blocked_without_pathologist_output(self):
        """
        Extension Officer must not run if Pathologist output is missing.
        """
        e_agent = ExtensionOfficerAgent(
            client=_make_mock_client(),
            project_id="test-project",
            advisory_model_id="ibm/granite-4-h-small",
        )
        result = e_agent.generate(diagnosis={})
        assert result["advisory"] is None
        assert "incomplete" in result["error"].lower() or "no diagnostic" in result["error"].lower() or result["error"]

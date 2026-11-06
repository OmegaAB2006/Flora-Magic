"""
ibm_client.py — IBM watsonx.ai authentication, model initialization,
vision inference, and text inference.

Never log or expose credentials.
"""

import base64
import json
import logging
import time
from io import BytesIO
from typing import Optional

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def create_ibm_client(api_key: str, service_url: str) -> APIClient:
    """
    Authenticate with IBM watsonx.ai and return an APIClient instance.
    Raises ValueError on missing credentials.
    Raises ibm_watsonx_ai exceptions on auth failure — let callers handle them.
    """
    if not api_key or not service_url:
        raise ValueError("IBM API key and service URL are required.")

    credentials = Credentials(
        url=service_url,
        api_key=api_key,
    )
    client = APIClient(credentials)
    return client


def validate_connection(client: APIClient, project_id: str) -> dict:
    """
    Validate that the client can reach watsonx.ai and access the project.
    Returns {"ok": True} or {"ok": False, "error": "..."}.
    """
    try:
        # A lightweight check — list foundation models metadata
        client.foundation_models.TextModels  # attribute access; does not call API
        # Actually ping by listing available models
        _ = client.foundation_models.get_model_specs()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Vision inference — Pathologist Agent
# ---------------------------------------------------------------------------

def analyze_crop_image(
    client: APIClient,
    project_id: str,
    vision_model_id: str,
    image_bytes: bytes,
    crop_name: str,
    symptoms: str,
    location: str,
    max_new_tokens: int = 1024,
    timeout: int = 120,
) -> dict:
    """
    Call an IBM Granite Vision model with the supplied image and context.

    Returns a dict with keys:
        raw_text   — raw model response string
        usage      — token usage dict (or empty dict)
        latency    — seconds elapsed
    Raises on API errors — callers should catch.
    """
    # Build the multimodal message
    crop_context = crop_name.strip() if crop_name else "unknown crop"
    symptom_context = symptoms.strip() if symptoms else "No additional symptoms described."
    location_context = location.strip() if location else "Not specified."

    system_prompt = (
        "You are an expert plant pathologist AI assistant. "
        "Analyze the provided crop image carefully and return ONLY a valid JSON object "
        "matching the schema described in the user message. "
        "Do not add markdown fences, commentary, or text outside the JSON object."
    )

    user_text = f"""Analyze this crop image and return a JSON diagnostic report.

Crop: {crop_context}
Location: {location_context}
Farmer-reported symptoms: {symptom_context}

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "agent": "Pathologist Agent",
  "image_quality": "<acceptable|poor|unacceptable>",
  "crop_name": "<identified or provided crop name>",
  "suspected_condition": "<brief description of suspected condition>",
  "condition_category": "<e.g. Possible fungal disease, Bacterial disease, Pest damage, Nutrient deficiency, Environmental stress, Physical damage, Unclear>",
  "observed_symptoms": ["<symptom 1>", "<symptom 2>"],
  "possible_causes": ["<cause 1>", "<cause 2>"],
  "alternative_diagnoses": ["<alternative 1>"],
  "confidence_level": "<strong visual indications|moderate visual indications|insufficient evidence>",
  "requires_expert_review": true,
  "status": "<preliminary_assessment|insufficient_image_quality|unclear>"
}}

Rules:
- If the image is blurry, too dark, or the plant/symptoms are not visible, set image_quality to "unacceptable" and status to "insufficient_image_quality".
- Do NOT invent conditions you cannot see in the image.
- Always set requires_expert_review to true.
- confidence_level must use only the three phrases listed above.
"""

    # Encode image to base64
    encoded_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]

    model = ModelInference(
        model_id=vision_model_id,
        api_client=client,
        project_id=project_id,
        params={
            "max_new_tokens": max_new_tokens,
        },
    )

    t0 = time.monotonic()
    response = model.chat(messages=messages, max_new_tokens=max_new_tokens)
    latency = time.monotonic() - t0

    raw_text = ""
    usage = {}

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})
    else:
        raw_text = str(response)

    return {"raw_text": raw_text, "usage": usage, "latency": latency}


# ---------------------------------------------------------------------------
# Text inference — Extension Officer Agent
# ---------------------------------------------------------------------------

def generate_farmer_advisory(
    client: APIClient,
    project_id: str,
    advisory_model_id: str,
    diagnosis: dict,
    max_new_tokens: int = 1024,
    timeout: int = 120,
) -> dict:
    """
    Call IBM Granite 4 H Small with the validated diagnosis dict.

    Returns a dict with keys:
        raw_text   — raw model response string
        usage      — token usage dict (or empty dict)
        latency    — seconds elapsed
    Raises on API errors — callers should catch.
    """
    diagnosis_json = json.dumps(diagnosis, indent=2)

    system_prompt = (
        "You are an experienced agricultural extension officer AI assistant. "
        "Convert the diagnostic JSON into practical, safe, farmer-friendly advice. "
        "Return ONLY a valid JSON object matching the schema in the user message. "
        "Do not add markdown fences or commentary outside the JSON object."
    )

    user_text = f"""Generate a farmer advisory based on this crop diagnosis:

{diagnosis_json}

Return ONLY this JSON structure:
{{
  "agent": "Extension Officer Agent",
  "crop_health_summary": "<explain the suspected problem in simple language>",
  "immediate_actions": ["<low-risk action 1>", "<low-risk action 2>"],
  "preventive_measures": ["<prevention 1>", "<prevention 2>"],
  "crop_care_recommendations": ["<care recommendation 1>", "<care recommendation 2>"],
  "treatment_guidance": "<general guidance only; recommend consulting a local expert before applying any pesticide>",
  "expert_consultation": "<when and why to contact an agricultural extension officer>",
  "safety_warning": "Follow all local agricultural regulations and product safety labels before applying any treatment."
}}

Rules:
- Do NOT invent pesticide names, dosages, concentrations, or application intervals.
- If the diagnosis confidence is low, provide only conservative low-risk guidance.
- Always recommend expert consultation.
- Reflect the uncertainty described in the diagnosis.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    model = ModelInference(
        model_id=advisory_model_id,
        api_client=client,
        project_id=project_id,
        params={
            "max_new_tokens": max_new_tokens,
        },
    )

    t0 = time.monotonic()
    response = model.chat(messages=messages, max_new_tokens=max_new_tokens)
    latency = time.monotonic() - t0

    raw_text = ""
    usage = {}

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})
    else:
        raw_text = str(response)

    return {"raw_text": raw_text, "usage": usage, "latency": latency}

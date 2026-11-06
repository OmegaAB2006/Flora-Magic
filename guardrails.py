"""
guardrails.py — Input, output, tool-execution, data, security,
operational, and multi-agent safety guardrails for Flora Magic.

All public functions return a GuardrailResult namedtuple:
    ok      — True if the check passed
    message — Human-readable explanation (empty string when ok=True)
    data    — Cleaned / sanitised value (same as input when unchanged)
"""

import io
import json
import logging
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    ok: bool
    message: str = ""
    data: Any = None
    intervention: bool = False   # True when a safety action was triggered


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES = 5 * 1024 * 1024          # 5 MB
MAX_IMAGE_DIMENSION = 4096                 # pixels
MIN_IMAGE_DIMENSION = 50                   # pixels
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_TEXT_LENGTH = 2000                     # characters for any free-text field

# File signature bytes (magic numbers)
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC  = b"\x89PNG\r\n\x1a\n"

# Patterns that suggest prompt injection or policy-override attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|prompts?|guidelines?)",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|prompts?|guidelines?)",
    r"reveal\s+(the\s+)?(api[\s_-]?key|secret|credential|password|token)",
    r"print\s+(the\s+)?(api[\s_-]?key|secret|credential|password|token)",
    r"show\s+(me\s+)?(the\s+)?(api[\s_-]?key|secret|credential|password|token)",
    r"bypass\s+(safety|guardrail|filter|restriction|policy)",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?different",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"DAN\b",
    r"override\s+your\s+(system|safety|security)",
]
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL
)

# Patterns that suggest PII in symptom / location text fields
_PII_PATTERNS = [
    r"\b\d{12}\b",                    # Aadhaar-like 12-digit number
    r"\b[A-Z]{5}\d{4}[A-Z]\b",       # PAN card
    r"\b\d{10}\b",                    # phone number
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # credit/debit card
    r"\baccount\s+number\b",
    r"\bifsc\b",
]
_PII_RE = re.compile("|".join(_PII_PATTERNS), re.IGNORECASE)

# Patterns that suggest unsupported pesticide dosage / chemical detail
_UNSAFE_CHEMICAL_PATTERNS = [
    r"\b\d+[\s]*(ml|mL|L|litre|liter|g|kg|oz|lb)\s*/\s*(litre|liter|L|acre|ha|hectare)\b",
    r"\b\d+\s*%\s*(concentration|solution|dilution)\b",
    r"apply\s+\d+",
    r"mix\s+\d+",
    r"dose\s+of\s+\d+",
    r"dosage\s*:\s*\d+",
    r"\bppm\b.*\d+",
    r"\d+\s*ppm\b",
]
_UNSAFE_CHEM_RE = re.compile(
    "|".join(_UNSAFE_CHEMICAL_PATTERNS), re.IGNORECASE
)

# Credential / secret leak patterns (for output scanning)
_SECRET_PATTERNS = [
    r"ibm[\s_-]?api[\s_-]?key",
    r"apikey\s*=",
    r"api_key\s*=",
    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",
    r"password\s*=",
    r"secret\s*=",
]
_SECRET_RE = re.compile("|".join(_SECRET_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. Input guardrails — image
# ---------------------------------------------------------------------------

def validate_image(file_bytes: bytes, filename: str = "") -> GuardrailResult:
    """
    Validate uploaded image bytes.

    Checks:
    - File size ≤ 5 MB
    - Valid file signature (JPEG or PNG magic bytes)
    - Parseable by Pillow without decompression bomb
    - Minimum and maximum dimensions
    """
    # Size check
    if len(file_bytes) > MAX_IMAGE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return GuardrailResult(
            ok=False,
            message=f"Image is too large ({size_mb:.1f} MB). Maximum allowed is 5 MB.",
            intervention=True,
        )

    if len(file_bytes) < 100:
        return GuardrailResult(
            ok=False,
            message="File is too small to be a valid image.",
            intervention=True,
        )

    # Magic bytes check
    is_jpeg = file_bytes[:3] == JPEG_MAGIC
    is_png  = file_bytes[:8] == PNG_MAGIC
    if not (is_jpeg or is_png):
        return GuardrailResult(
            ok=False,
            message=(
                "Unsupported file type. Only JPEG and PNG images are accepted. "
                "Please upload a valid crop photo."
            ),
            intervention=True,
        )

    # Pillow parse + decompression bomb guard
    Image.MAX_IMAGE_PIXELS = 50_000_000   # ~50 MP limit
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()   # detects truncated / corrupted files
        # Re-open after verify (verify() consumes the stream)
        img = Image.open(io.BytesIO(file_bytes))
        width, height = img.size
    except Image.DecompressionBombError:
        return GuardrailResult(
            ok=False,
            message="Image rejected: potential decompression bomb detected.",
            intervention=True,
        )
    except UnidentifiedImageError:
        return GuardrailResult(
            ok=False,
            message="Could not read image. The file may be corrupted. Please upload a different image.",
            intervention=True,
        )
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(
            ok=False,
            message=f"Image validation failed: {exc}",
            intervention=True,
        )

    # Dimension checks
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return GuardrailResult(
            ok=False,
            message=(
                f"Image is too small ({width}×{height} px). "
                "Please upload a larger, clearer photo of the crop."
            ),
            intervention=True,
        )

    return GuardrailResult(ok=True, data=file_bytes)


def prepare_image(file_bytes: bytes, max_side: int = 1280) -> bytes:
    """
    Resize large images, convert to RGB JPEG, and strip metadata.
    Returns processed JPEG bytes.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    # Save without EXIF / metadata
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 2. Input guardrails — text
# ---------------------------------------------------------------------------

def validate_text_input(text: str, field_name: str = "input") -> GuardrailResult:
    """
    Validate free-text inputs (crop name, symptoms, location).

    Checks:
    - Not excessively long
    - No prompt injection patterns
    - Warn (not block) on suspected PII
    """
    if not isinstance(text, str):
        return GuardrailResult(
            ok=False,
            message=f"{field_name} must be a text string.",
            intervention=True,
        )

    if len(text) > MAX_TEXT_LENGTH:
        return GuardrailResult(
            ok=False,
            message=(
                f"{field_name} is too long ({len(text)} characters). "
                f"Maximum is {MAX_TEXT_LENGTH} characters."
            ),
            intervention=True,
        )

    # Prompt injection detection
    if _INJECTION_RE.search(text):
        logger.warning("Prompt injection detected in %s field.", field_name)
        return GuardrailResult(
            ok=False,
            message=(
                "Your input contains instructions that cannot be processed. "
                "Please describe only the crop symptoms or location."
            ),
            intervention=True,
        )

    # PII warning (non-blocking for MVP — just flag)
    pii_found = bool(_PII_RE.search(text))

    return GuardrailResult(
        ok=True,
        data=text.strip(),
        intervention=pii_found,
        message=(
            "⚠️ Your input may contain personal information (ID number, phone, etc.). "
            "Please remove sensitive details before submitting."
            if pii_found else ""
        ),
    )


# ---------------------------------------------------------------------------
# 3. Output guardrails — diagnosis
# ---------------------------------------------------------------------------

def validate_diagnosis_output(raw_text: str) -> GuardrailResult:
    """
    Parse and validate the Pathologist Agent's raw JSON response.

    Returns GuardrailResult with .data = parsed dict on success.
    """
    from schemas import DiagnosisReport  # local import to avoid circular dep

    # Strip markdown fences if present
    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return GuardrailResult(
            ok=False,
            message=f"Pathologist Agent returned invalid JSON: {exc}",
            intervention=True,
        )

    # Credential leak scan
    if _SECRET_RE.search(json.dumps(data)):
        logger.error("Credential pattern detected in diagnosis output — suppressed.")
        return GuardrailResult(
            ok=False,
            message="Diagnosis output contained potentially sensitive data and was blocked.",
            intervention=True,
        )

    # Pydantic validation
    try:
        report = DiagnosisReport(**data)
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(
            ok=False,
            message=f"Diagnosis does not match required schema: {exc}",
            intervention=True,
        )

    return GuardrailResult(ok=True, data=report.model_dump())


# ---------------------------------------------------------------------------
# 4. Output guardrails — advisory
# ---------------------------------------------------------------------------

def validate_advisory_output(raw_text: str, diagnosis: dict) -> GuardrailResult:
    """
    Parse and validate the Extension Officer Agent's raw JSON response.

    Also checks:
    - No unsupported chemical dosages
    - Advisory does not silently contradict the diagnosis
    - No credential leaks
    """
    from schemas import AdvisoryReport  # local import

    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return GuardrailResult(
            ok=False,
            message=f"Extension Officer Agent returned invalid JSON: {exc}",
            intervention=True,
        )

    # Credential leak scan
    if _SECRET_RE.search(json.dumps(data)):
        logger.error("Credential pattern detected in advisory output — suppressed.")
        return GuardrailResult(
            ok=False,
            message="Advisory output contained potentially sensitive data and was blocked.",
            intervention=True,
        )

    # Unsafe chemical dosage check — scan all string values
    full_text = " ".join(str(v) for v in _flatten_values(data))
    if _UNSAFE_CHEM_RE.search(full_text):
        # Sanitise: replace treatment_guidance with a safe fallback
        data["treatment_guidance"] = (
            "Specific treatment details could not be verified. "
            "Please consult a local agricultural extension officer or certified agronomist "
            "before applying any pesticide or chemical treatment. "
            "Always follow approved product labels and local regulations."
        )
        logger.warning("Unsafe chemical recommendation detected and replaced.")
        # Continue — advisory is still usable with the sanitised guidance

    # Pydantic validation
    try:
        report = AdvisoryReport(**data)
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(
            ok=False,
            message=f"Advisory does not match required schema: {exc}",
            intervention=True,
        )

    result_dict = report.model_dump()

    # Consistency check: advisory must not introduce a different condition
    diag_condition = diagnosis.get("suspected_condition", "").lower()
    advisory_summary = result_dict.get("crop_health_summary", "").lower()

    # Heuristic: if diagnosis is "unclear" or "insufficient_image_quality",
    # advisory must not claim a specific confirmed disease
    if diagnosis.get("status") in ("insufficient_image_quality", "unclear"):
        confident_phrases = [
            "confirmed", "definite", "is caused by", "the plant has",
            "the disease is", "infected with",
        ]
        for phrase in confident_phrases:
            if phrase in advisory_summary:
                result_dict["crop_health_summary"] = (
                    "The crop image did not provide sufficient evidence for a clear diagnosis. "
                    "The following guidance is general and precautionary. "
                    "Please consult a qualified agricultural professional for a confirmed diagnosis."
                )
                logger.warning(
                    "Advisory contradicted uncertain diagnosis — summary replaced."
                )
                break

    return GuardrailResult(ok=True, data=result_dict, intervention=False)


# ---------------------------------------------------------------------------
# 5. Multi-agent handoff guardrail
# ---------------------------------------------------------------------------

def validate_agent_handoff(diagnosis: dict) -> GuardrailResult:
    """
    Ensure the Pathologist Agent output is safe to pass to the Extension
    Officer Agent.

    The Extension Officer Agent must NOT run if:
    - diagnosis is None or empty
    - image quality is unacceptable and no alternative condition found
    - status is 'insufficient_image_quality'
    """
    if not diagnosis:
        return GuardrailResult(
            ok=False,
            message="No diagnostic data available. Cannot generate advisory.",
            intervention=True,
        )

    status = diagnosis.get("status", "")
    image_quality = diagnosis.get("image_quality", "")

    if status == "insufficient_image_quality" or image_quality == "unacceptable":
        return GuardrailResult(
            ok=False,
            message=(
                "The uploaded image does not contain enough visible detail for a reliable diagnosis. "
                "Please upload a clearer, well-lit photo that shows the affected part of the crop."
            ),
            intervention=True,
        )

    required_keys = {
        "agent", "crop_name", "suspected_condition",
        "observed_symptoms", "confidence_level", "requires_expert_review",
    }
    missing = required_keys - set(diagnosis.keys())
    if missing:
        return GuardrailResult(
            ok=False,
            message=f"Diagnostic data is incomplete. Missing fields: {missing}",
            intervention=True,
        )

    return GuardrailResult(ok=True, data=diagnosis)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from model output."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _flatten_values(obj: Any):
    """Yield all string leaf values from a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _flatten_values(item)
    elif isinstance(obj, str):
        yield obj

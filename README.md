# 🌿 Flora Magic — AI-Powered Crop Health Diagnosis & Farmer Advisory

Flora Magic is a multi-agent AI application built with Python, Streamlit, and IBM watsonx.ai.
It analyses crop images with an IBM Granite Vision model (Pathologist Agent) and generates
farmer-friendly advisory reports using IBM Granite 4 H Small (Extension Officer Agent).

---

## Features

- Upload a crop image (JPG/JPEG/PNG, up to 5 MB)
- Pathologist Agent analyses the image using IBM Granite Vision
- Extension Officer Agent generates safe, practical agricultural advice
- Input, output, and multi-agent guardrails throughout
- Download the final advisory report as a `.txt` file
- Agent Performance dashboard with execution metrics

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd flora-Magic
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM credentials

**Option A — Streamlit secrets (recommended for local use)**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and fill in your real credentials
```

**Option B — Environment variables**

```bash
export IBM_API_KEY="your-ibm-cloud-api-key"
export IBM_PROJECT_ID="your-watsonx-project-id"
export IBM_SERVICE_URL="https://us-south.ml.cloud.ibm.com"
export VISION_MODEL_ID="ibm/granite-vision-3-3-2b"
export ADVISORY_MODEL_ID="ibm/granite-4-h-small"
```

**Option C — Sidebar input**

Enter credentials directly in the Streamlit sidebar when the app is running.
They are used only for the current session and never stored.

### 5. Run the application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## IBM watsonx.ai Setup

1. Create an [IBM Cloud account](https://cloud.ibm.com/).
2. Create a **watsonx.ai** project and note the **Project ID**.
3. Generate an **IBM Cloud API key** from IAM.
4. Confirm that the following models are available in your region:
   - `ibm/granite-vision-3-3-2b` — vision model for the Pathologist Agent
   - `ibm/granite-4-h-small` — text model for the Extension Officer Agent
5. Use the **Test Connection** button in the sidebar to verify.

> If a model is not available in your region, enter the deployment ID of a
> compatible IBM vision model in the **Vision Model ID** field.

---

## Project Structure

```
flora-Magic/
├── app.py              # Streamlit UI and workflow orchestration
├── agents.py           # Pathologist Agent and Extension Officer Agent
├── ibm_client.py       # IBM watsonx.ai authentication and inference
├── guardrails.py       # Input, output, and multi-agent safety checks
├── schemas.py          # Pydantic models for agent outputs
├── utils.py            # Image prep, report generation, metrics helpers
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml.example   # Credential template (never commit the real file)
└── tests/
    ├── test_guardrails.py      # Guardrail unit tests
    └── test_workflow.py        # Agent handoff integration tests
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use mocked IBM API responses and do not require real credentials.

---

## Safety and Guardrails

Flora Magic implements the following safety controls:

| Guardrail | Description |
|---|---|
| Image validation | File size, magic bytes, dimensions, decompression bomb check |
| Text validation | Length limits, prompt injection detection, PII warning |
| Output validation | Pydantic schema enforcement, credential leak scan |
| Unsafe chemical filter | Blocks unsupported pesticide dosages from advisory |
| Multi-agent handoff | Extension Officer cannot run without valid Pathologist output |
| Credential protection | API keys never logged or displayed |
| Uncertainty enforcement | Uncertain diagnoses cannot be presented as confirmed |

---

## Disclaimer

Flora Magic generates AI-powered **preliminary visual assessments only**.
Results are **not** confirmed laboratory diagnoses.
Always consult a qualified agricultural professional before making
significant treatment or management decisions.

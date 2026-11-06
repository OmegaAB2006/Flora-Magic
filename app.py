"""
app.py — Flora Magic: AI-Powered Crop Health Diagnosis & Farmer Advisory
Main Streamlit application.

Run with:
    streamlit run app.py
"""

import logging
import os
import sys
import time

import streamlit as st

# ── Ensure the project directory is on sys.path ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import agents as agent_module
import guardrails as gr
import ibm_client as ibm
import utils
from schemas import SessionMetrics

# ---------------------------------------------------------------------------
# Logging — INFO level; never logs credentials
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s: %(message)s",
)
logger = logging.getLogger("flora_magic")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flora Magic",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# FLORAL THEME — injected once at the top
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* ── Base background — soft petal white ─────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #fdf6ff 0%, #f0faf0 50%, #fff8f0 100%);
    background-attachment: fixed;
}

/* ── Sidebar — deep botanical green ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a4a2e 0%, #2d6a4f 60%, #1b5e3b 100%) !important;
    border-right: 3px solid #52b788;
}
section[data-testid="stSidebar"] * {
    color: #d8f3dc !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid #74c69d !important;
    color: #d8f3dc !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stToggle label {
    color: #b7e4c7 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #95d5b2 !important;
    border-bottom: 1px solid #52b788 !important;
    padding-bottom: 4px !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #52b788 !important;
    opacity: 0.5 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #52b788, #74c69d) !important;
    color: #1a4a2e !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    padding: 6px 18px !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #74c69d, #95d5b2) !important;
    transform: translateY(-1px) !important;
}

/* ── Tab bar ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.6) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 1px solid #b7e4c7 !important;
    gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 22px !important;
    font-weight: 700 !important;
    color: #2d6a4f !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #52b788, #40916c) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(64,145,108,0.35) !important;
}

/* ── Hero banner ─────────────────────────────────────────────────────────── */
.flora-hero {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 40%, #40916c 70%, #52b788 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(27,67,50,0.25);
}
.flora-hero::before {
    content: "🌸 🌿 🌺 🌱 🌼 🍃 🌻 🌾 🌷";
    position: absolute;
    top: 8px; right: 16px;
    font-size: 22px;
    opacity: 0.25;
    letter-spacing: 6px;
    line-height: 1.8;
    white-space: nowrap;
    overflow: hidden;
    width: 60%;
    text-align: right;
}
.flora-hero h1 {
    color: #d8f3dc !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    margin-bottom: 6px !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    letter-spacing: -0.5px;
}
.flora-hero p {
    color: #b7e4c7 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    max-width: 560px;
}
.flora-hero .flora-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #d8f3dc !important;
    margin-top: 14px;
    backdrop-filter: blur(4px);
}

/* ── Section cards ───────────────────────────────────────────────────────── */
.flora-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid #b7e4c7;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(64,145,108,0.08);
    backdrop-filter: blur(6px);
}
.flora-card h3 {
    color: #1b4332 !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
}

/* ── Result summary cards ────────────────────────────────────────────────── */
.flora-result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(240,250,240,0.95));
    border: 2px solid #74c69d;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px rgba(64,145,108,0.12);
}
.flora-result-card .label {
    font-size: 0.8rem;
    color: #40916c;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.flora-result-card .value {
    font-size: 1rem;
    color: #1b4332;
    font-weight: 600;
    line-height: 1.5;
}

/* ── Disclaimer banner ───────────────────────────────────────────────────── */
.flora-disclaimer {
    background: linear-gradient(135deg, #fff3cd, #ffeeba);
    border-left: 5px solid #f6c23e;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 18px 0;
    font-size: 0.88rem;
    color: #856404;
    box-shadow: 0 2px 8px rgba(246,194,62,0.15);
}

/* ── Primary action button ───────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #40916c, #52b788) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 36px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(64,145,108,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: linear-gradient(135deg, #2d6a4f, #40916c) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(64,145,108,0.45) !important;
}
.stButton > button[kind="primary"]:disabled {
    opacity: 0.45 !important;
}

/* ── Download button ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #e9f5ee, #d8f3dc) !important;
    color: #1b4332 !important;
    border: 2px solid #74c69d !important;
    border-radius: 25px !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7) !important;
    transform: translateY(-1px) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: linear-gradient(90deg, #e9f5ee, #f0faf0) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    color: #1b4332 !important;
    border: 1px solid #b7e4c7 !important;
}
.streamlit-expanderContent {
    border: 1px solid #b7e4c7 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: rgba(255,255,255,0.85) !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid #b7e4c7 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    box-shadow: 0 2px 8px rgba(64,145,108,0.08) !important;
}
[data-testid="stMetricLabel"] {
    color: #40916c !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
[data-testid="stMetricValue"] {
    color: #1b4332 !important;
    font-weight: 800 !important;
}

/* ── Info / success / warning / error callouts ───────────────────────────── */
.stAlert[data-baseweb="notification"] {
    border-radius: 12px !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #74c69d !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.6) !important;
    padding: 12px !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    border: 1.5px solid #b7e4c7 !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.85) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #52b788 !important;
    box-shadow: 0 0 0 3px rgba(82,183,136,0.15) !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #b7e4c7 !important;
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #52b788, #74c69d) !important;
    border-radius: 10px !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border-color: #b7e4c7 !important;
    opacity: 0.6 !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0faf0; border-radius: 10px; }
::-webkit-scrollbar-thumb { background: #74c69d; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #40916c; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "metrics": utils.init_session_metrics(),
        "diagnosis": None,
        "advisory": None,
        "pathologist_exec": None,
        "extension_exec": None,
        "workflow_success": False,
        "workflow_ran": False,
        "ibm_client_obj": None,
        "last_image_hash": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

# ---------------------------------------------------------------------------
# Helper — load credentials from Streamlit secrets or environment
# ---------------------------------------------------------------------------
def _secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets first, then environment, then fallback."""
    try:
        return st.secrets.get(key, os.environ.get(key, fallback))
    except Exception:  # noqa: BLE001
        return os.environ.get(key, fallback)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("## 🌸 Flora Magic")
    st.caption("🌿 AI-Powered Crop Health Assistant")
    st.divider()

    # ── IBM Configuration ──────────────────────────────────────────────────
    st.markdown("### 🔑 IBM watsonx.ai Configuration")

    api_key = st.text_input(
        "IBM API Key",
        value=_secret("IBM_API_KEY"),
        type="password",
        help="Your IBM Cloud API key. Never shared or logged.",
    )
    project_id = st.text_input(
        "Project ID",
        value=_secret("IBM_PROJECT_ID"),
        help="IBM watsonx.ai project ID.",
    )
    service_url = st.text_input(
        "Service URL",
        value=_secret("IBM_SERVICE_URL", "https://us-south.ml.cloud.ibm.com"),
        help="IBM watsonx.ai service endpoint URL.",
    )
    vision_model_id = st.text_input(
        "Vision Model ID",
        value=_secret("VISION_MODEL_ID", "ibm/granite-vision-3-3-2b"),
        help="IBM Granite Vision model ID or deployment ID.",
    )
    advisory_model_id = st.text_input(
        "Advisory Model ID",
        value=_secret("ADVISORY_MODEL_ID", "ibm/granite-4-h-small"),
        help="IBM Granite 4 H Small model ID.",
    )

    # ── Connection status ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔗 Connection Status")

    configured = bool(api_key and project_id and service_url)

    if not configured:
        st.warning("⚠️ Not Configured — enter IBM credentials above.")
        st.session_state["ibm_client_obj"] = None
    else:
        if st.button("🔌 Test Connection"):
            with st.spinner("Connecting to IBM watsonx.ai…"):
                try:
                    client_obj = ibm.create_ibm_client(api_key, service_url)
                    check = ibm.validate_connection(client_obj, project_id)
                    if check["ok"]:
                        st.success("✅ Connected")
                        st.session_state["ibm_client_obj"] = client_obj
                    else:
                        st.error(f"Connection Failed: {check['error']}")
                        st.session_state["ibm_client_obj"] = None
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Connection Failed: {exc}")
                    st.session_state["ibm_client_obj"] = None
        else:
            if st.session_state["ibm_client_obj"] is not None:
                st.success("✅ Connected")
            else:
                st.info("ℹ️ Configured — click 'Test Connection' to verify.")

    # ── Application Settings ───────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Settings")
    show_agent_details = st.toggle("🔬 Show agent details", value=True)
    show_performance    = st.toggle("📊 Show performance metrics", value=True)
    max_tokens          = st.slider("Max output tokens", 512, 2048, 1024, 128)

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#74c69d;text-align:center;opacity:0.8'>"
        "🌸 Flora Magic v1.0 · Powered by IBM watsonx.ai"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# MAIN TABS
# ===========================================================================
tab_diagnosis, tab_performance = st.tabs(["🌾 Crop Diagnosis", "📊 Agent Performance"])


# ===========================================================================
# TAB 1 — CROP DIAGNOSIS
# ===========================================================================
with tab_diagnosis:

    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="flora-hero">
        <h1>🌸 FLORA MAGIC</h1>
        <p>Upload an image of your crop to identify possible health problems
        and receive practical, farmer-friendly agricultural guidance.</p>
        <span class="flora-badge">🤖 Powered by IBM Granite Vision &amp; Granite 4 H Small</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="flora-disclaimer">
        ⚠️ <strong>Disclaimer:</strong> AI-generated preliminary assessment.
        Results are <strong>not</strong> a confirmed laboratory diagnosis.
        Always consult a qualified agricultural professional before taking action.
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="flora-card"><h3>🌱 Crop Information</h3>', unsafe_allow_html=True)
        crop_name = st.text_input(
            "Crop Name",
            placeholder="e.g. Tomato, Rice, Potato, Chilli, Cotton",
            help="Enter the name of the crop you are growing.",
        )
        location = st.text_input(
            "📍 Location (optional)",
            placeholder="e.g. Punjab, Maharashtra, Tamil Nadu",
            help="Your state or region helps contextualise agricultural advice.",
        )
        symptoms = st.text_area(
            "🔍 Observed Symptoms (optional)",
            placeholder="e.g. Yellow spots on leaves and gradual drying.",
            height=110,
            help="Describe what you can see. This helps the AI focus its analysis.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="flora-card"><h3>📷 Upload Crop Image</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image (JPG / PNG, max 5 MB)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit photo of the affected crop or leaf.",
        )

        if uploaded_file:
            raw_bytes = utils.load_image_bytes(uploaded_file)
            img_check = gr.validate_image(raw_bytes, uploaded_file.name)
            if img_check.ok:
                st.image(
                    utils.image_to_display(raw_bytes),
                    caption="📸 Uploaded Crop Image",
                    use_container_width=True,
                )
            else:
                st.error(f"❌ {img_check.message}")
                uploaded_file = None
        else:
            st.markdown(
                "<div style='text-align:center;padding:30px 0;color:#74c69d;font-size:3rem;'>🌿</div>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Diagnose button ───────────────────────────────────────────────────
    btn_col, info_col = st.columns([1, 2])
    with btn_col:
        diagnose_clicked = st.button(
            "🔬 Diagnose Crop",
            type="primary",
            disabled=(not configured or uploaded_file is None),
            help="Configure IBM credentials and upload an image to enable.",
            use_container_width=True,
        )
    with info_col:
        if not configured:
            st.info("ℹ️ Enter your IBM watsonx.ai credentials in the sidebar to begin.")
        elif uploaded_file is None:
            st.info("📷 Upload a crop image to enable diagnosis.")

    # ── Workflow execution ────────────────────────────────────────────────
    if diagnose_clicked and uploaded_file and configured:

        # Reset previous results
        for k in ["diagnosis", "advisory", "pathologist_exec", "extension_exec"]:
            st.session_state[k] = None
        st.session_state["workflow_success"] = False
        st.session_state["workflow_ran"] = True

        raw_bytes = utils.load_image_bytes(uploaded_file)

        # Image guardrail
        img_validation = gr.validate_image(raw_bytes, uploaded_file.name)
        if not img_validation.ok:
            st.error(f"❌ Image rejected: {img_validation.message}")
            st.stop()

        # Text field guardrails
        for field_label, field_value in [
            ("Crop Name", crop_name),
            ("Location", location),
            ("Symptoms", symptoms),
        ]:
            if field_value:
                text_check = gr.validate_text_input(field_value, field_label)
                if not text_check.ok:
                    st.error(f"❌ {text_check.message}")
                    st.stop()
                if text_check.intervention and text_check.message:
                    st.warning(text_check.message)

        image_bytes = gr.prepare_image(raw_bytes)

        try:
            client_obj = ibm.create_ibm_client(api_key, service_url)
            st.session_state["ibm_client_obj"] = client_obj
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ IBM connection error: {exc}")
            st.stop()

        # ── AGENT 1: Pathologist ──────────────────────────────────────────
        with st.status("🌿 Analyzing crop image with Pathologist Agent…", expanded=True) as status_box:
            st.write("🔬 IBM Granite Vision is examining your crop photo…")

            pathologist = agent_module.PathologistAgent(
                client=client_obj,
                project_id=project_id,
                vision_model_id=vision_model_id,
                max_new_tokens=max_tokens,
            )
            p_result = pathologist.analyze(
                image_bytes=image_bytes,
                crop_name=crop_name,
                symptoms=symptoms,
                location=location,
            )

            st.session_state["pathologist_exec"] = p_result["execution"]

            if p_result["error"]:
                status_box.update(label="❌ Pathologist Agent failed.", state="error")
                st.error(f"Pathologist Agent: {p_result['error']}")
                st.session_state["metrics"] = utils.record_workflow_result(
                    st.session_state["metrics"], p_result["execution"], None, success=False,
                )
                st.stop()

            diagnosis = p_result["diagnosis"]
            st.session_state["diagnosis"] = diagnosis
            status_box.update(label="✅ Crop image analyzed.", state="complete")

        # ── Image quality check ───────────────────────────────────────────
        if diagnosis.get("image_quality") == "unacceptable" or diagnosis.get("status") == "insufficient_image_quality":
            st.warning(
                "📷 **Insufficient Image Quality**\n\n"
                "The Pathologist Agent could not identify enough visible detail. "
                "Please upload a clearer, well-lit photo that shows the affected part of the crop."
            )
            st.session_state["metrics"] = utils.record_workflow_result(
                st.session_state["metrics"], p_result["execution"], None, success=False,
            )
            st.stop()

        # ── AGENT 2: Extension Officer ────────────────────────────────────
        with st.status("🌾 Preparing farmer advisory with Extension Officer Agent…", expanded=True) as status_box2:
            st.write("📋 IBM Granite 4 H Small is generating your advisory…")

            extension_officer = agent_module.ExtensionOfficerAgent(
                client=client_obj,
                project_id=project_id,
                advisory_model_id=advisory_model_id,
                max_new_tokens=max_tokens,
            )
            e_result = extension_officer.generate(diagnosis=diagnosis)

            st.session_state["extension_exec"] = e_result["execution"]

            if e_result["error"]:
                status_box2.update(label="❌ Extension Officer Agent failed.", state="error")
                st.error(f"Extension Officer Agent: {e_result['error']}")
                st.session_state["metrics"] = utils.record_workflow_result(
                    st.session_state["metrics"], p_result["execution"], e_result["execution"], success=False,
                )
                st.stop()

            advisory = e_result["advisory"]
            st.session_state["advisory"] = advisory
            status_box2.update(label="✅ Agricultural advisory ready.", state="complete")

        st.session_state["workflow_success"] = True
        st.session_state["metrics"] = utils.record_workflow_result(
            st.session_state["metrics"], p_result["execution"], e_result["execution"], success=True,
        )

    # ── Results display ───────────────────────────────────────────────────
    if st.session_state.get("workflow_success") and st.session_state.get("diagnosis"):
        diagnosis = st.session_state["diagnosis"]
        advisory  = st.session_state["advisory"]

        st.divider()
        st.success("🌸 Diagnosis and advisory completed successfully!")

        # ── Pathologist Agent report ──────────────────────────────────────
        if show_agent_details:
            with st.expander("🔬 Pathologist Agent — Diagnostic Findings", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("🌿 Crop", diagnosis.get("crop_name", "N/A"))
                col2.metric("📷 Image Quality", diagnosis.get("image_quality", "N/A").capitalize())
                col3.metric("📊 Confidence", diagnosis.get("confidence_level", "N/A").split()[0].capitalize())

                col4, col5, col6 = st.columns(3)
                col4.metric("🏷️ Condition Category", diagnosis.get("condition_category", "N/A"))
                col5.metric("📋 Status", diagnosis.get("status", "N/A").replace("_", " ").capitalize())
                col6.metric("👨‍⚕️ Expert Review", "Yes ✓" if diagnosis.get("requires_expert_review") else "No")

                st.markdown("**🌡️ Suspected Condition:**")
                st.info(diagnosis.get("suspected_condition", "N/A"))

                scol, ccol = st.columns(2)
                with scol:
                    st.markdown("**🍂 Observed Symptoms:**")
                    for s in diagnosis.get("observed_symptoms", []):
                        st.markdown(f"🔸 {s}")
                with ccol:
                    st.markdown("**⚗️ Possible Causes:**")
                    for c in diagnosis.get("possible_causes", []):
                        st.markdown(f"🔹 {c}")

                if diagnosis.get("alternative_diagnoses"):
                    st.markdown("**🔄 Alternative Diagnoses:**")
                    for a in diagnosis["alternative_diagnoses"]:
                        st.markdown(f"◦ {a}")

        # ── Extension Officer advisory ────────────────────────────────────
        if show_agent_details:
            with st.expander("🌾 Extension Officer Agent — Farmer Advisory", expanded=True):
                st.markdown("**🌱 Crop Health Summary:**")
                st.info(advisory.get("crop_health_summary", "N/A"))

                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown("**⚡ Immediate Actions:**")
                    for a in advisory.get("immediate_actions", []):
                        st.markdown(f"🔸 {a}")
                    st.markdown("**🛡️ Preventive Measures:**")
                    for p in advisory.get("preventive_measures", []):
                        st.markdown(f"🔹 {p}")
                with ac2:
                    st.markdown("**🌿 Crop-Care Recommendations:**")
                    for r in advisory.get("crop_care_recommendations", []):
                        st.markdown(f"🍃 {r}")

                st.markdown("**💊 Treatment Guidance:**")
                st.warning(advisory.get("treatment_guidance", "Consult a local agricultural expert."))
                st.markdown("**👨‍🌾 When to Consult an Expert:**")
                st.markdown(advisory.get("expert_consultation", "N/A"))
                st.markdown("**⚠️ Safety Warning:**")
                st.error(advisory.get("safety_warning", "Follow local regulations."))

        # ── Final report summary cards ────────────────────────────────────
        st.divider()
        st.markdown("### 📄 Flora Magic — Crop Health Report")

        r1, r2, r3 = st.columns(3)
        actions = advisory.get("immediate_actions", [])
        with r1:
            st.markdown(f"""
            <div class="flora-result-card">
                <div class="label">🌡️ Suspected Condition</div>
                <div class="value">{diagnosis.get("suspected_condition", "N/A")}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="flora-result-card">
                <div class="label">⚡ Top Immediate Action</div>
                <div class="value">{actions[0] if actions else "See full advisory."}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="flora-result-card">
                <div class="label">👨‍🌾 Expert Consultation</div>
                <div class="value">{advisory.get("expert_consultation", "Contact your local extension officer.")[:120]}…</div>
            </div>""", unsafe_allow_html=True)

        # Report text for download
        report_text = utils.build_text_report(
            diagnosis=diagnosis,
            advisory=advisory,
            crop_name=crop_name if "crop_name" in dir() else "",
            location=location if "location" in dir() else "",
        )
        with st.expander("📜 View Full Report Text", expanded=False):
            st.text(report_text)

        st.divider()
        # Disclaimer
        st.markdown("""
        <div class="flora-disclaimer">
            📋 <strong>Disclaimer:</strong> This report is AI-generated and represents a preliminary visual
            assessment. It is <strong>not</strong> a confirmed laboratory diagnosis. Consult a qualified
            agricultural professional before making significant treatment decisions.
            Flora Magic is a decision-support tool and does not replace expert agronomic advice.
        </div>
        """, unsafe_allow_html=True)

        # Download
        st.download_button(
            label="📥 Download Advisory Report (.txt)",
            data=report_text,
            file_name="flora_magic_crop_report.txt",
            mime="text/plain",
        )

    elif st.session_state.get("workflow_ran") and not st.session_state.get("workflow_success"):
        st.info("The previous diagnosis workflow did not complete successfully. Check the messages above.")


# ===========================================================================
# TAB 2 — AGENT PERFORMANCE DASHBOARD
# ===========================================================================
with tab_performance:
    st.markdown("""
    <div class="flora-hero" style="padding:28px 40px;margin-bottom:20px">
        <h1 style="font-size:1.9rem!important">📊 Agent Performance Dashboard</h1>
        <p>Session-level metrics — refreshed after each diagnosis run.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics: SessionMetrics = st.session_state["metrics"]

    if not show_performance:
        st.info("Performance metrics are hidden. Enable them in the sidebar.")
    else:
        # ── Top-level metrics ─────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌱 Total Requests", metrics.total_requests)
        m2.metric("✅ Successful Workflows", metrics.successful_workflows)
        m3.metric("❌ Failed Workflows", metrics.failed_workflows)
        m4.metric("🛡️ Guardrail Interventions", metrics.total_guardrail_interventions)

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("📡 API Calls Made", metrics.total_api_calls)
        m6.metric("📥 Input Tokens", metrics.total_input_tokens if metrics.total_input_tokens else "—")
        m7.metric("📤 Output Tokens", metrics.total_output_tokens if metrics.total_output_tokens else "—")
        m8.metric("💰 Est. Cost", "Cost data unavailable.")

        avg_lat = utils.average_latency(metrics)
        st.metric("⏱️ Average Workflow Latency", f"{avg_lat:.2f}s" if avg_lat else "—")

        # ── Per-agent latency breakdown ───────────────────────────────────
        p_exec = st.session_state.get("pathologist_exec")
        e_exec = st.session_state.get("extension_exec")

        if p_exec or e_exec:
            st.markdown("#### ⏱️ Last Workflow Latency")
            lc1, lc2 = st.columns(2)
            p_lat = p_exec.latency_seconds if p_exec else 0.0
            e_lat = e_exec.latency_seconds if e_exec else 0.0
            lc1.metric("🔬 Pathologist Agent", f"{p_lat:.2f}s" if p_exec else "—")
            lc2.metric("🌾 Extension Officer Agent", f"{e_lat:.2f}s" if e_exec else "—")
            total_lat = p_lat + e_lat
            if total_lat > 0:
                st.progress(
                    p_lat / total_lat,
                    text=f"🔬 Pathologist: {p_lat:.2f}s  |  🌾 Extension Officer: {e_lat:.2f}s  |  Total: {total_lat:.2f}s",
                )

        # ── Execution log table ───────────────────────────────────────────
        st.markdown("#### 📋 Agent Execution Log")
        rows = utils.execution_log_rows(metrics)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("🌿 No execution data yet. Run a diagnosis to populate this table.")

        # ── Latency history chart ─────────────────────────────────────────
        completed_execs = [e for e in metrics.executions if e.status == "completed"]
        if len(completed_execs) >= 2:
            st.markdown("#### 📈 Latency History")
            import pandas as pd
            df_chart = pd.DataFrame({
                "Run": range(1, len(completed_execs) + 1),
                "Latency (s)": [e.latency_seconds for e in completed_execs],
            }).set_index("Run")
            st.line_chart(df_chart, color="#52b788")

        # ── Reset ─────────────────────────────────────────────────────────
        st.divider()
        if st.button("🔄 Reset Session Metrics"):
            for k in ["metrics", "diagnosis", "advisory", "pathologist_exec", "extension_exec"]:
                st.session_state[k] = utils.init_session_metrics() if k == "metrics" else None
            st.session_state["workflow_success"] = False
            st.session_state["workflow_ran"] = False
            st.rerun()

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from classifier import PhishGuardClassifier
from llm.explainer import PhishGuardExplainer

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "ml" / "artifacts"
CURRENT = ARTIFACTS / "current"
DATA = ROOT / "data"
LOG_FILE = ROOT / "logs" / "pipeline.log"
PID_FILE = ROOT / "logs" / "pipeline.pid"

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: #0b1020; color: #e8edf8; }
[data-testid="stSidebar"] { background: #111a2e; }
.hero {
    padding: 1.4rem 1.6rem; border-radius: 16px;
    background: linear-gradient(120deg,#152746,#172033);
    border: 1px solid #2a3b5c; margin-bottom: 1rem;
}
.muted { color: #9eacc5; }
.metric-card {
    padding: 1rem; border: 1px solid #2a3b5c;
    background: #111a2e; border-radius: 12px;
}
.verdict {
    padding: 1rem 1.2rem; border-radius: 12px;
    border: 1px solid #40516f; background: #111a2e;
    margin: .5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def model_files_present() -> bool:
    required = [
        "catboost.joblib", "lightgbm.joblib",
        "xgboost.joblib", "meta_learner.joblib",
    ]
    return all((CURRENT / name).is_file() for name in required)

@st.cache_resource
def load_classifier():
    return PhishGuardClassifier()

def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

def pipeline_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError, ProcessLookupError):
        PID_FILE.unlink(missing_ok=True)
        return False

def start_pipeline(loop: bool):
    (ROOT / "logs").mkdir(exist_ok=True)
    log_handle = LOG_FILE.open("a", encoding="utf-8")
    args = [sys.executable, str(ROOT / "pipeline_runner.py")]
    if loop:
        args += ["--loop"]
    process = subprocess.Popen(
        args, cwd=ROOT, stdout=log_handle,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    PID_FILE.write_text(str(process.pid))
    st.session_state["pipeline_process"] = process
    st.session_state["pipeline_log_handle"] = log_handle

def stop_pipeline():
    process = st.session_state.get("pipeline_process")
    stopped = False
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
        stopped = True
    if not stopped and PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except (ValueError, OSError, ProcessLookupError):
            pass
    PID_FILE.unlink(missing_ok=True)

st.markdown("""
<div class="hero">
  <h1 style="margin:0">🛡️ PhishGuard</h1>
  <p style="font-size:1.05rem;margin:.35rem 0 0 0">
    Local-first URL phishing analysis · Threat intelligence · ML ensemble · LLM explanations
  </p>
  <p class="muted" style="margin:.35rem 0 0 0">
    Laptop prototype — model inference and Ollama explanations run locally.
    PhishHunt lookups require an internet connection.
  </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("PhishGuard")
    page = st.radio(
        "Workspace",
        ["Analyze URL", "Data & Training", "System status", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Prototype build · Local ML artifacts")

if page == "Analyze URL":
    st.subheader("URL analysis")
    st.write("Submit a URL for a PhishHunt lookup, ensemble classification, and evidence-based explanation.")
    url = st.text_input(
        "URL",
        placeholder="https://example.com/login",
        help="Only HTTP and HTTPS URLs are accepted. The submitted URL is sent to PhishHunt for threat-intelligence analysis.",
    )
    c1, c2 = st.columns([1, 4])
    with c1:
        analyze = st.button("Analyze URL", type="primary", use_container_width=True)
    with c2:
        st.caption("Do not submit private URLs containing access tokens or sensitive query parameters.")

    if analyze:
        if not url.strip():
            st.error("Enter a URL first.")
        elif not model_files_present():
            st.error("No trained model set is installed. Open Data & Training and run the training pipeline first.")
        else:
            try:
                with st.spinner("Checking threat intelligence and running the ML ensemble..."):
                    classifier = load_classifier()
                    result = classifier.classify(url.strip())
                with st.spinner("Preparing the security explanation..."):
                    report = PhishGuardExplainer().explain(result)

                st.session_state["last_report"] = report.model_dump()
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    report_data = st.session_state.get("last_report")
    if report_data:
        prediction = report_data["final_prediction"]
        known = report_data["threat_intelligence"]["status"] == "KNOWN_PHISHING"
        st.markdown("---")
        st.subheader("Security report")
        verdict_color = "#6b1f2b" if prediction == "PHISHING" else "#4a3b18"
        st.markdown(
            f"""<div class="verdict" style="background:{verdict_color}22">
            <div class="muted">FINAL CLASSIFICATION</div>
            <h2 style="margin:.2rem 0">{prediction.replace("_", " ")}</h2>
            <div>Source: {report_data['classification_source'].replace('_',' ').title()}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        ti = report_data["threat_intelligence"]
        ml = report_data["machine_learning"]
        a, b, c = st.columns(3)
        a.metric("PhishHunt", ti["status"].replace("_", " "))
        b.metric("ML phishing score", f"{ml['phishing_score']:.4f}" if ml else "Skipped")
        c.metric("Explanation", report_data["explanation_source"].replace("_", " "))

        if ti.get("verdict"):
            st.caption(f"PhishHunt heuristic verdict: {ti['verdict']}")
        if ti["status"] == "UNAVAILABLE":
            st.warning("PhishHunt could not be reached. The ML result was used without a successful threat-intelligence lookup.")
        elif ti["status"] == "NOT_FOUND":
            st.info("PhishHunt reported no known threat match. This does not mean the URL is safe.")
        elif known:
            st.error("PhishHunt reported a known threat match. The ML stage was skipped.")

        st.markdown("### Explanation")
        st.write(report_data["explanation"])

        if ml:
            st.markdown("### Model outputs")
            scores = ml["base_model_scores"]
            cols = st.columns(len(scores) or 1)
            for col, (name, score) in zip(cols, scores.items()):
                col.metric(name.title(), f"{score:.4f}")
            st.caption(f"Decision threshold: {ml['threshold']:.2f}. The ensemble score is a model output, not necessarily a calibrated real-world probability.")

        left, right = st.columns(2)
        with left:
            st.markdown("### Evidence")
            for item in report_data["evidence"]:
                st.markdown(f"- {item}")
        with right:
            st.markdown("### Limitations")
            for item in report_data["limitations"]:
                st.markdown(f"- {item}")

        st.markdown("### Recommended actions")
        for item in report_data["recommendations"]:
            st.markdown(f"- {item}")

        with st.expander("Raw structured report"):
            st.json(report_data)

elif page == "Data & Training":
    st.subheader("Data acquisition and model training")
    st.write("The pipeline downloads the PhishHunt feed and a Tranco candidate-domain sample, extracts URL features, then trains and evaluates the stacking ensemble.")
    st.warning("Training can be CPU-intensive and may take a while, especially during the first run when Optuna searches hyperparameters.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PhishHunt feed", "Present" if (DATA / "phishunt_feed.json").exists() else "Missing")
    with col2:
        st.metric("Tranco candidates", "Present" if (DATA / "benign_urls.json").exists() else "Missing")
    with col3:
        st.metric("Current model set", "Ready" if model_files_present() else "Missing")

    is_running = pipeline_running()
    st.markdown("#### Pipeline controls")
    x, y, z = st.columns(3)
    with x:
        if st.button("Run pipeline once", type="primary", disabled=is_running, use_container_width=True):
            start_pipeline(loop=False)
            st.rerun()
    with y:
        if st.button("Start hourly refresh", disabled=is_running, use_container_width=True):
            start_pipeline(loop=True)
            st.rerun()
    with z:
        if st.button("Stop pipeline", disabled=not is_running, use_container_width=True):
            stop_pipeline()
            st.rerun()

    st.caption("Hourly mode runs a complete acquisition → feature extraction → training → evaluation cycle, then waits until the next hourly interval.")
    if is_running:
        st.success("Pipeline process is running.")
    else:
        st.info("Pipeline process is not running.")

    if LOG_FILE.exists():
        st.markdown("#### Pipeline log")
        log_text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        st.code(log_text[-14000:] or "No log output yet.", language="text")
        if st.button("Refresh log"):
            st.rerun()

    evaluation = load_json(CURRENT / "evaluation.json")
    if evaluation:
        st.markdown("#### Current model evaluation")
        metric_cols = st.columns(4)
        for col, key, label in zip(
            metric_cols,
            ["accuracy", "precision", "recall", "roc_auc"],
            ["Accuracy", "Precision", "Recall", "ROC-AUC"],
        ):
            value = evaluation.get(key)
            col.metric(label, f"{value:.4f}" if isinstance(value, (float, int)) else "N/A")
        st.caption(f"Evaluation samples: {evaluation.get('test_samples', 'N/A')}")

elif page == "System status":
    st.subheader("System status")
    st.write("Local components and expected runtime dependencies.")
    checks = {
        "Model artifacts": model_files_present(),
        "PhishHunt feed cache": (DATA / "phishunt_feed.json").exists(),
        "Tranco candidate cache": (DATA / "benign_urls.json").exists(),
        "Feature dataset": (DATA / "processed" / "url_features.json").exists(),
        "Ollama API": False,
    }
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        checks["Ollama API"] = response.ok
        if response.ok:
            tags = response.json().get("models", [])
            st.caption("Ollama models: " + (", ".join(x.get("name", "") for x in tags) or "No models reported"))
    except Exception:
        pass

    for name, ok in checks.items():
        st.write(("🟢" if ok else "⚪") + f"  {name}: {'Ready' if ok else 'Not detected'}")

    st.caption("Ollama being unavailable does not block classification; the explainer uses its deterministic Python fallback.")

else:
    st.subheader("About this prototype")
    st.markdown("""
    **Three integrated subsystems**

    1. **Data acquisition:** PhishHunt phishing feed + Tranco ranked-domain candidates.
    2. **Machine learning:** CatBoost, LightGBM and XGBoost base models with a Logistic Regression stacking meta-learner.
    3. **LLM explanation:** local Qwen through Ollama. Python owns the verdict; the LLM only writes an explanation.

    **Classification flow**

    - PhishHunt known-threat match → `PHISHING`; ML is skipped.
    - No known match → run the ML ensemble.
    - PhishHunt unavailable → run ML and clearly report the intelligence outage.
    - An ML result of `BENIGN_CANDIDATE` is not a safety certification.

    **Important limitations**

    - Tranco is a popularity-ranked domain list, not a verified-benign ground-truth dataset.
    - Evaluation against PhishHunt-vs-Tranco labels can contain source and collection bias.
    - The model score is not automatically a calibrated probability.
    - This is a research/laptop prototype, not an enterprise-grade browser protection product.
    """)

# PhishGuard — Laptop Prototype

A local-first research prototype combining:

1. **Data acquisition:** PhishHunt phishing feed + Tranco ranked-domain candidates.
2. **ML classification:** CatBoost + LightGBM + XGBoost stacked by Logistic Regression.
3. **LLM explanation:** Qwen2.5 through local Ollama; Python remains authoritative for the verdict.

## 1. Requirements

- Ubuntu/Linux
- Python 3.10+ (Python 3.11/3.12 recommended for smoother ML-wheel compatibility)
- Internet access for feed acquisition and PhishHunt lookups
- Ollama installed for local LLM explanations (optional; Python fallback works without it)
- Enough RAM/disk for CatBoost, LightGBM, XGBoost and Optuna

## 2. Install

```bash
cd PhishGuard_Laptop
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

If a wheel is unavailable for your Python version, use Python 3.11 or 3.12 and recreate the virtual environment.

## 3. Ollama (optional but recommended)

```bash
ollama pull qwen2.5:0.5b
ollama serve
```

Keep Ollama running in another terminal. If Ollama is unavailable, PhishGuard still classifies URLs and emits a deterministic Python explanation.

## 4. First-time data acquisition and training

```bash
source .venv/bin/activate
python pipeline_runner.py
```

The first cycle downloads data, extracts features, runs Optuna hyperparameter search if no saved parameters exist, trains the base models and meta-learner, evaluates the candidate, and promotes it to `ml/artifacts/current`.

Initial optimization may take a while. Later cycles reuse `best_params.json`.

## 5. Launch the UI

```bash
streamlit run app.py
```

Or:

```bash
./run.sh
```

Streamlit normally opens at `http://localhost:8501`.

## 6. Using the app

- **Analyze URL:** submit an HTTP/HTTPS URL and inspect the threat-intelligence result, ensemble outputs, explanation, evidence and limitations.
- **Data & Training:** run a single pipeline cycle or start hourly refresh.
- **System status:** check model files, cached datasets and Ollama.
- **About:** review the classification flow and limitations.

## 7. Classification policy

- A PhishHunt known-threat flag produces `PHISHING` and skips ML.
- `NOT_FOUND` means no known PhishHunt match was returned; it does not mean safe.
- `UNAVAILABLE` means the intelligence request failed; ML still runs.
- `BENIGN_CANDIDATE` is not a verified-safe verdict.
- ML scores are model outputs, not automatically calibrated real-world probabilities.
- The LLM is explanation-only. It cannot change the Python-generated classification.

## 8. Important research limitations

- Tranco is a popularity ranking, not a verified-benign dataset.
- Training labels are source-derived (PhishHunt vs Tranco); source artifacts and temporal drift can inflate apparent evaluation quality.
- The prototype does not fetch or render the submitted website. PhishHunt's analysis endpoint and URL features are used.
- This is not a substitute for browser isolation, endpoint protection, reputation services, or human review.
- Do not treat a `BENIGN_CANDIDATE` result as permission to enter credentials or payment details.

## 9. Project structure

```text
PhishGuard_Laptop/
├── app.py
├── pipeline_runner.py
├── classifier.py
├── threat_intelligence.py
├── feature_extractor.py
├── live_feature_extractor.py
├── phish_url_collector.py
├── benign_url_collector.py
├── optimize.py
├── train.py
├── meta_learner.py
├── evaluate.py
├── llm/
│   ├── explainer.py
│   ├── llm_client.py
│   ├── prompts.py
│   └── schemas.py
├── data/
└── ml/artifacts/
```

## 10. Current prototype boundaries

The uploaded source files did not include your trained model binaries, feed caches, or processed dataset. You must run the initial pipeline on your laptop to generate those local artifacts. This archive contains the integrated application and source code, not pre-trained model weights.

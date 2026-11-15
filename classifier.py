from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from live_feature_extractor import extract_live_features
from threat_intelligence import check_phishunt

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "ml" / "artifacts" / "current"

FEATURE_COLUMNS = [
    "count_subdomains",
    "ip_address",
    "at_symbol",
    "url_length",
    "url_depth",
    "redirection_double_slash",
    "http_https_in_domain",
    "https_scheme",
    "url_shortener",
    "prefix_suffix_hyphen",
    "contains_sensitive_word",
    "contains_trendy_brand",
    "uppercase_letter",
    "number_of_dots",
]

MODEL_FILES = {
    "catboost": "catboost.joblib",
    "lightgbm": "lightgbm.joblib",
    "xgboost": "xgboost.joblib",
}


class PhishGuardClassifier:

    def __init__(self):
        self.models = {}
        self.meta_learner = None
        self.load_models()

    def load_models(self):
        for name, filename in MODEL_FILES.items():
            model_path = MODEL_DIR / filename

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Missing model: {model_path}"
                )

            self.models[name] = joblib.load(model_path)

        meta_path = MODEL_DIR / "meta_learner.joblib"

        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Missing meta-learner: {meta_path}"
            )

        self.meta_learner = joblib.load(meta_path)

        print("[PhishGuard] All ML models loaded.")

    @staticmethod
    def validate_url(url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL cannot be empty.")

        url = url.strip()

        if len(url) > 2048:
            raise ValueError("URL exceeds 2048 characters.")

        if "://" not in url:
            url = "http://" + url

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are supported.")

        if not parsed.hostname:
            raise ValueError("URL hostname is missing.")

        try:
            parsed.port
        except ValueError as error:
            raise ValueError("Invalid URL port.") from error

        return url

    def classify(self, url: str, threshold: float = 0.5) -> dict:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        url = self.validate_url(url)

        # -----------------------------------------
        # STAGE 1: THREAT INTELLIGENCE
        # -----------------------------------------

        intelligence = check_phishunt(url)

        if intelligence["status"] == "KNOWN_PHISHING":
            return {
                "url": url,
                "prediction": "PHISHING",
                "classification_source": "THREAT_INTELLIGENCE",
                "phishunt": intelligence,
                "ml": None,
            }

        # -----------------------------------------
        # STAGE 2: MACHINE LEARNING
        # -----------------------------------------

        features = extract_live_features(url)

        X = pd.DataFrame(
            [[features[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        base_scores = {}

        for name, model in self.models.items():
            probabilities = model.predict_proba(X)

            classes = list(model.classes_)
            phishing_index = classes.index(1)

            base_scores[name] = float(
                probabilities[0][phishing_index]
            )

        meta_input = np.array([[
            base_scores["catboost"],
            base_scores["lightgbm"],
            base_scores["xgboost"],
        ]], dtype=float)

        final_probabilities = self.meta_learner.predict_proba(
            meta_input
        )

        meta_classes = list(self.meta_learner.classes_)
        phishing_index = meta_classes.index(1)

        phishing_score = float(
            final_probabilities[0][phishing_index]
        )

        prediction = (
            "PHISHING"
            if phishing_score >= threshold
            else "BENIGN_CANDIDATE"
        )

        return {
            "url": url,
            "prediction": prediction,
            "classification_source": "MACHINE_LEARNING",
            "phishunt": intelligence,
            "ml": {
                "phishing_score": phishing_score,
                "benign_score": 1.0 - phishing_score,
                "threshold": threshold,
                "base_model_scores": base_scores,
                "features": features,
            },
        }

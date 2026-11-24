#!/usr/bin/env python3
"""Train and evaluate PhishGuard's URL classifier.

The historical filename is retained for project compatibility. This version
uses character n-gram TF-IDF + URL/hostname features + Logistic Regression.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


def normalize_url(value: object) -> str:
    """Tolerant, idempotent URL canonicalisation; never rejects a nonblank URL."""
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return ""
    candidate = raw if "://" in raw else "http://" + raw.lstrip("/")
    try:
        parsed = urlsplit(candidate)
        host = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
    except ValueError:
        return raw
    if not host:
        return raw
    host = host.rstrip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    result = host + (f":{port}" if port is not None else "") + (parsed.path or "/")
    if parsed.query:
        result += "?" + parsed.query
    if parsed.fragment:
        result += "#" + parsed.fragment
    return result


def _parse_for_features(value: object):
    text = normalize_url(value)
    candidate = text if "://" in text else "http://" + text
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        parsed = urlsplit("")
        host = ""
    return text, parsed, host


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = np.fromiter((text.count(ch) for ch in set(text)), dtype=float)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Numerical features that complement, rather than replace, URL text."""
    feature_names = [
        "url_length", "host_length", "path_length", "query_length",
        "host_dot_count", "host_hyphen_count", "host_digit_count",
        "host_alpha_count", "subdomain_count", "path_depth",
        "query_parameter_count", "at_symbol_count", "percent_count",
        "equals_count", "ampersand_count", "underscore_count",
        "hyphen_count", "digit_ratio", "special_char_ratio",
        "host_entropy", "url_entropy", "is_ip_address", "has_punycode",
        "tld_length", "suspicious_word_count", "longest_label",
        "double_slash_in_path", "has_ip_like_text",
    ]
    suspicious_words = (
        "login", "signin", "verify", "account", "update", "secure",
        "bank", "wallet", "password", "confirm", "webscr", "paypal",
        "support", "recover", "billing", "invoice", "auth",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for value in X:
            text, parsed, host = _parse_for_features(value)
            path, query = parsed.path or "", parsed.query or ""
            labels = [part for part in host.split(".") if part]
            tld = labels[-1] if labels else ""
            total = max(len(text), 1)
            digits = sum(ch.isdigit() for ch in text)
            special = sum(not ch.isalnum() for ch in text)
            try:
                import ipaddress
                ipaddress.ip_address(host)
                is_ip = 1.0
            except ValueError:
                is_ip = 0.0
            lowered = text.lower()
            suspicious = sum(lowered.count(word) for word in self.suspicious_words)
            row = [
                len(text), len(host), len(path), len(query),
                host.count("."), host.count("-"), sum(c.isdigit() for c in host),
                sum(c.isalpha() for c in host), max(len(labels) - 2, 0),
                len([p for p in path.split("/") if p]),
                query.count("&") + (1 if query else 0), text.count("@"),
                text.count("%"), text.count("="), text.count("&"),
                text.count("_"), text.count("-"), digits / total,
                special / total, _entropy(host), _entropy(text), is_ip,
                float("xn--" in host), len(tld), suspicious,
                max((len(label) for label in labels), default=0),
                float("//" in path), float(bool(re.search(r"\d{1,3}(?:\.\d{1,3}){3}", text))),
            ]
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


# Ensure joblib records a stable importable module path even when this file is
# executed as a script (where Python would otherwise record __main__).
if __name__ == "__main__":
    sys.modules.setdefault("mlp_model", sys.modules[__name__])
URLFeatureExtractor.__module__ = "mlp_model"


def build_model(max_features: int = 300000, min_df: int = 2, c_value: float = 2.0):
    features = FeatureUnion(
        transformer_list=[
            ("char_tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=min_df,
                max_features=max_features, sublinear_tf=True,
                dtype=np.float32, lowercase=True,
            )),
            ("url_features", Pipeline([
                ("extract", URLFeatureExtractor()),
                ("scale", StandardScaler(with_mean=False)),
            ])),
        ],
        n_jobs=None,
    )
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            solver="liblinear", C=c_value, max_iter=1000,
            class_weight="balanced", random_state=42,
        )),
    ])


def load_split(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run dataset_manager.py first.")
    frame = pd.read_csv(path)
    required = {"url", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    urls = [normalize_url(value) for value in frame["url"].fillna("").astype(str)]
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{path} contains missing or non-numeric labels.")
    y = labels.astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{path} labels must be 0 (legitimate) or 1 (phishing).")
    if any(not url for url in urls):
        bad_count = sum(not url for url in urls)
        raise ValueError(f"{path} contains {bad_count} blank URLs after normalisation.")
    if len(np.unique(y)) != 2:
        raise ValueError(f"{path} must contain both classes; found {np.unique(y).tolist()}.")
    return np.asarray(urls, dtype=object), y


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, max_fpr: float) -> float:
    """Choose the lowest threshold whose empirical validation FPR <= max_fpr."""
    if not 0.0 <= max_fpr < 1.0:
        raise ValueError("--max-fpr must be in [0, 1).")
    benign_scores = np.sort(probabilities[y_true == 0])
    if len(benign_scores) == 0:
        raise ValueError("Validation split has no legitimate examples.")
    candidates = np.unique(np.concatenate(([0.0], benign_scores, [1.0])))
    valid = []
    for threshold in candidates:
        fpr = float(np.mean(benign_scores >= threshold))
        if fpr <= max_fpr:
            valid.append(float(threshold))
    if not valid:
        return float(np.nextafter(1.0, 2.0))
    return min(valid)


def calculate_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PhishGuard URL model.")
    parser.add_argument("--artifacts", type=Path, default=Path("phishguard_artifacts"))
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--max-features", type=int, default=300000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=2.0)
    args = parser.parse_args()

    print("[1/5] Loading domain-separated data...")
    X_train, y_train = load_split(args.artifacts / "train.csv", "train")
    X_val, y_val = load_split(args.artifacts / "validation.csv", "validation")
    X_test, y_test = load_split(args.artifacts / "test.csv", "test")
    print(f"      Train={len(y_train):,}, validation={len(y_val):,}, test={len(y_test):,}")

    print("[2/5] Building model...")
    model = build_model(args.max_features, args.min_df, args.c)
    print("[3/5] Training...")
    model.fit(X_train, y_train)

    print("[4/5] Calibrating threshold on validation data...")
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob, args.max_fpr)
    val_metrics = calculate_metrics(y_val, val_prob, threshold)

    print("[5/5] Evaluating once on held-out test data...")
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_prob, threshold)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts / "phishguard_model.joblib"
    joblib.dump({"model": model, "threshold": threshold, "label_order": [0, 1]}, model_path)

    report = {
        "model": "character TF-IDF + URL features + LogisticRegression",
        "label_mapping": {"0": "legitimate", "1": "phishing"},
        "threshold_selection": f"validation FPR <= {args.max_fpr}",
        "max_validation_fpr_target": args.max_fpr,
        "validation": val_metrics,
        "test": test_metrics,
        "classification_report_test": classification_report(
            y_test, (test_prob >= threshold).astype(int),
            labels=[0, 1], target_names=["legitimate", "phishing"],
            output_dict=True, zero_division=0,
        ),
        "notes": [
            "The FPR constraint is calibrated on validation data, not guaranteed on future traffic.",
            "The test set is evaluated once after threshold selection.",
            "The Tranco list is popularity-based, not a safety guarantee.",
        ],
    }
    (args.artifacts / "model_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame([
        {"split": "validation", **val_metrics},
        {"split": "test", **test_metrics},
    ]).to_csv(args.artifacts / "model_metrics.csv", index=False)

    print(f"      Selected threshold: {threshold:.6f}")
    print(f"      Validation FPR: {val_metrics['fpr']:.4%}")
    print(f"      Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"      Test precision: {test_metrics['precision']:.4f}")
    print(f"      Test recall: {test_metrics['recall']:.4f}")
    print(f"      Test F1: {test_metrics['f1']:.4f}")
    print(f"      Test ROC-AUC: {test_metrics['roc_auc']}")
    print(f"      Model saved: {model_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

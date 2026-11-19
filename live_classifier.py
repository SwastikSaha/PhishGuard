#!/usr/bin/env python3
"""
PhishGuard command-line URL classifier.

Dependencies:
    pip install torch pandas numpy scikit-learn joblib

Expected files:
    mlp_model.py              (the MLP implementation you will create separately)
    phishguard_mlp.pth        (trained model state_dict)
    phishguard_artifacts/
        preprocessor.joblib   (created by dataset_manager.py)

The mlp_model.py file must expose:
    class URLClassifier(nn.Module):
        def __init__(self, input_size, hidden_sizes=(256, 128), num_classes=2)
        def forward(self, x)

IMPORTANT:
The feature extraction rules below are reasonable approximations based on
the dataset column descriptions. For reliable inference, they MUST match
the exact feature-generation logic used to create the training CSV.
"""

from __future__ import annotations

import argparse
import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import torch

# This import will work once you create the separate MLP implementation file.
from mlp_model import URLClassifier


SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "verification", "update",
    "secure", "account", "banking", "password", "confirm",
    "wallet", "invoice", "payment", "recover", "suspend",
    "unlock", "authenticate",
}


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def parse_url(raw_url: str):
    raw_url = raw_url.strip()
    candidate = raw_url if "://" in raw_url else "http://" + raw_url
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").lower().strip(".")
    return raw_url, parsed, host


def extract_features(raw_url: str) -> dict:
    """
    Build the 14 numeric + 1 categorical features expected by the
    saved scikit-learn preprocessor.
    """
    original, parsed, host = parse_url(raw_url)
    if not host:
        raise ValueError("Could not extract a hostname from this URL.")

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = [part for part in host.split(".") if part]
    tld = labels[-1] if labels else "unknown"

    # Approximation: assume the registrable domain is the final two labels.
    # Multi-part public suffixes (e.g. co.uk) need a public-suffix-aware parser.
    subdomain_count = max(0, len(labels) - 2) if not is_ip else 0

    lowered_url = original.lower()
    keyword_flag = int(
        any(re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered_url)
            for word in SUSPICIOUS_KEYWORDS)
    )

    return {
        "url_length": len(original),
        "url_entropy": shannon_entropy(original),
        "slash_count": original.count("/"),
        "query_length": len(parsed.query),
        "domain_length": len(host),
        "domain_entropy": shannon_entropy(host),
        "sub_domain": subdomain_count,
        "digit_count": sum(ch.isdigit() for ch in original),
        "special_char_count": sum(not ch.isalnum() for ch in original),
        "hyphen_count": original.count("-"),
        "https_flag": int(parsed.scheme.lower() == "https"),
        "ip_flag": is_ip,
        "at_flag": int("@" in original),
        "keyword_flag": keyword_flag,
        "tld": tld,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a URL with PhishGuard.")
    parser.add_argument("url", nargs="?", help="URL to classify.")
    parser.add_argument("--artifacts", default="phishguard_artifacts")
    parser.add_argument("--model", default="phishguard_mlp.pth")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Phishing probability threshold (default: 0.5).")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    url = args.url or input("Enter URL: ").strip()
    if not url:
        raise ValueError("No URL supplied.")

    artifact_dir = Path(args.artifacts)
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    model_path = Path(args.model)

    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    preprocessor = joblib.load(preprocessor_path)

    features = extract_features(url)
    feature_frame = pd.DataFrame([features])

    # The preprocessor applies the same imputing/scaling/one-hot encoding
    # fitted on the training partition.
    transformed = preprocessor.transform(feature_frame).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = URLClassifier(
        input_size=transformed.shape[1],
        hidden_sizes=(256, 128),
        num_classes=2,
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tensor = torch.from_numpy(transformed).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        phishing_probability = float(probabilities[1].item())
        legitimate_probability = float(probabilities[0].item())

    prediction = (
        "PHISHING" if phishing_probability >= args.threshold
        else "LEGITIMATE"
    )

    print("\n========== PHISHGUARD RESULT ==========")
    print(f"URL:                    {url}")
    print(f"Prediction:             {prediction}")
    print(f"Legitimate probability: {legitimate_probability:.4f}")
    print(f"Phishing probability:   {phishing_probability:.4f}")
    print(f"Threshold:              {args.threshold:.2f}")
    print("\nNote: This is a model prediction, not a guarantee of website safety.")


if __name__ == "__main__":
    main()

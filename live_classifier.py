
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import torch

from mlp_model import URLClassifier


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

NUMERIC_FEATURES = [
    "url_length", "url_entropy", "slash_count", "query_length",
    "domain_length", "domain_entropy", "sub_domain", "digit_count",
    "special_char_count", "hyphen_count", "https_flag", "ip_flag",
    "at_flag", "keyword_flag",
]

SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "verification", "update", "secure",
    "account", "banking", "password", "confirm", "wallet", "invoice",
    "payment", "recover", "suspend", "unlock", "authenticate",
}

# These are transparent heuristic observations, NOT feature attributions.
KEYWORD_PATTERN = re.compile(
    r"(?<![a-z])(" + "|".join(map(re.escape, sorted(SUSPICIOUS_KEYWORDS))) +
    r")(?![a-z])",
    re.IGNORECASE,
)


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length)
                for count in counts.values())


def parse_url(raw_url: str):
    raw_url = raw_url.strip()
    candidate = raw_url if "://" in raw_url else "http://" + raw_url
    parsed = urlsplit(candidate)

    # Reject malformed inputs early. Accessing .port can raise ValueError.
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid URL port: {exc}") from exc

    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        raise ValueError("Could not extract a hostname. Supply a URL such as https://example.com")
    return raw_url, parsed, host


def extract_features(raw_url: str) -> dict:
    original, parsed, host = parse_url(raw_url)

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    labels = [part for part in host.split(".") if part]
    tld = labels[-1] if labels else "unknown"
    subdomain_count = max(0, len(labels) - 2) if not is_ip else 0

    lowered_url = original.lower()
    keyword_flag = int(KEYWORD_PATTERN.search(lowered_url) is not None)

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


def analyse_observations(raw_url: str, features: dict) -> dict:
    observations = []
    reassuring = []

    if features["ip_flag"]:
        observations.append("The hostname is an IP address rather than a conventional domain name.")
    if features["at_flag"]:
        observations.append("The URL contains an '@' character, which can obscure the intended destination in some URL formats.")
    if features["keyword_flag"]:
        matches = sorted(set(m.group(0).lower() for m in KEYWORD_PATTERN.finditer(raw_url)))
        observations.append("The URL contains potentially sensitive terms: " + ", ".join(matches) + ".")
    if features["sub_domain"] >= 3:
        observations.append(f"The hostname has {features['sub_domain']} subdomain labels under the provisional parser.")
    if features["hyphen_count"] >= 3:
        observations.append(f"The URL contains {features['hyphen_count']} hyphens.")
    if features["url_length"] >= 100:
        observations.append(f"The URL is relatively long ({features['url_length']} characters).")
    if features["query_length"] >= 50:
        observations.append(f"The query string is relatively long ({features['query_length']} characters).")
    if features["https_flag"]:
        reassuring.append("The URL uses HTTPS. This encrypts the connection but does not prove the site is legitimate.")
    else:
        observations.append("The URL does not use HTTPS.")

    if not observations:
        observations.append("No indicators from the small set of configured URL heuristics were triggered.")
    if not reassuring:
        reassuring.append("No positive trust or reputation checks were performed.")

    return {
        "observed_indicators": observations,
        "reassuring_observations": reassuring,
        "limitations": [
            "These are URL-string observations, not proof of malicious activity.",
            "No DNS, certificate, redirect-chain, page-content, domain-age, or reputation lookup was performed.",
            "Feature extraction is provisional because the original dataset-generation code is unavailable.",
        ],
    }


def load_classifier(artifact_dir: Path, model_path: Path, requested_device: str):
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    config_path = artifact_dir / "model_config.json"

    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    preprocessor = joblib.load(preprocessor_path)
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}

    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    hidden_sizes = tuple(config.get("hidden_sizes", [256, 128]))

    return preprocessor, config, device, hidden_sizes


def predict_url(url: str, preprocessor, config: dict, device,
                hidden_sizes: tuple[int, int], model_path: Path):
    features = extract_features(url)
    feature_frame = pd.DataFrame([features])

    transformed = preprocessor.transform(feature_frame).astype(np.float32)
    expected_size = config.get("input_size")
    if expected_size is not None and transformed.shape[1] != int(expected_size):
        raise ValueError(
            f"Preprocessor produced {transformed.shape[1]} features, "
            f"but model_config.json expects {expected_size}."
        )

    model = URLClassifier(
        input_size=transformed.shape[1],
        hidden_sizes=hidden_sizes,
        num_classes=2,
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tensor = torch.from_numpy(transformed).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    return features, float(probabilities[0]), float(probabilities[1])


def generate_qwen_explanation(payload: dict, model_id: str,
                              max_new_tokens: int, llm_device: str) -> str:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is not installed. Run: "
            "python -m pip install transformers accelerate"
        ) from exc

    device = "cuda" if llm_device == "auto" and torch.cuda.is_available() else (
        "cpu" if llm_device == "auto" else llm_device
    )
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"\nLoading explanation model: {model_id} ({device}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    llm.eval()

    system_prompt = (
        "You are PhishGuard's security explanation component. "
        "The supplied classifier verdict and probabilities are authoritative and must never be changed. "
        "Explain only the supplied observations. Do not claim that a URL is confirmed malicious or safe. "
        "Do not invent domain reputation, ownership, DNS records, certificate status, redirects, page contents, "
        "or model feature attributions. Explicitly distinguish URL observations from the classifier's prediction. "
        "HTTPS is not proof of legitimacy. If evidence is limited, say so. "
        "Give a concise explanation understandable to a non-expert, followed by practical precautions."
    )

    user_prompt = (
        "Explain this PhishGuard analysis. The probabilities come from an experimental URL classifier. "
        "Feature extraction may differ from the original training feature-generation process.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        output = llm.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][encoded["input_ids"].shape[-1]:]
    explanation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Release model memory after one explanation.
    del llm
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return explanation


def display_report(url: str, prediction: str, legitimate_probability: float,
                   phishing_probability: float, threshold: float,
                   features: dict, evidence: dict, explanation: str | None):
    print("\n" + "=" * 62)
    print("                    PHISHGUARD REPORT")
    print("=" * 62)
    print(f"URL:                    {url}")
    print(f"Verdict:                {prediction}")
    print(f"Legitimate probability: {legitimate_probability:.2%}")
    print(f"Phishing probability:   {phishing_probability:.2%}")
    print(f"Decision threshold:     {threshold:.2%}")

    print("\n--- URL FEATURE OBSERVATIONS ---")
    for item in evidence["observed_indicators"]:
        print(f"[!] {item}")
    for item in evidence["reassuring_observations"]:
        print(f"[+] {item}")

    print("\n--- EXTRACTED FEATURES (PROVISIONAL) ---")
    for name in NUMERIC_FEATURES:
        print(f"{name:22s}: {features[name]}")
    print(f"{'tld':22s}: {features['tld']}")

    if explanation:
        print("\n--- QWEN EXPLANATION ---")
        print(explanation)

    print("\n--- LIMITATIONS ---")
    for item in evidence["limitations"]:
        print(f"- {item}")
    print("\nThe LLM explains the classifier output; it does not determine the verdict.")
    print("A model prediction is not a guarantee of website safety.")
    print("=" * 62)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a URL with PhishGuard and optionally explain it with Qwen."
    )
    parser.add_argument("url", nargs="?", help="URL to classify.")
    parser.add_argument("--artifacts", default="phishguard_artifacts")
    parser.add_argument("--model", default="phishguard_artifacts/phishguard_mlp.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="MLP device: auto or cpu (or supported torch device).")
    parser.add_argument("--no-explanation", action="store_true", help="Skip Qwen and print classifier/evidence only.")
    parser.add_argument("--llm", default=MODEL_ID, help="Hugging Face model ID for explanation.")
    parser.add_argument("--llm-device", default="auto", help="Qwen device: auto or cpu (or supported torch device).")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")

    url = args.url or input("Enter URL: ").strip()
    if not url:
        raise ValueError("No URL supplied.")

    artifact_dir = Path(args.artifacts).expanduser().resolve()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    preprocessor, config, device, hidden_sizes = load_classifier(
        artifact_dir, model_path, args.device
    )

    features, legitimate_probability, phishing_probability = predict_url(
        url, preprocessor, config, device, hidden_sizes, model_path
    )
    prediction = "PHISHING" if phishing_probability >= args.threshold else "LEGITIMATE"
    evidence = analyse_observations(url, features)

    payload = {
        "url": url,
        "verdict": prediction,
        "legitimate_probability": round(legitimate_probability, 6),
        "phishing_probability": round(phishing_probability, 6),
        "threshold": args.threshold,
        "features": features,
        "observed_indicators": evidence["observed_indicators"],
        "reassuring_observations": evidence["reassuring_observations"],
        "limitations": evidence["limitations"],
    }

    explanation = None
    if not args.no_explanation:
        try:
            explanation = generate_qwen_explanation(
                payload, args.llm, args.max_new_tokens, args.llm_device
            )
        except Exception as exc:
            print(f"\n[!] Qwen explanation unavailable: {exc}", file=sys.stderr)
            print("[!] Continuing with the MLP verdict and deterministic observations.", file=sys.stderr)

    display_report(
        url, prediction, legitimate_probability, phishing_probability,
        args.threshold, features, evidence, explanation
    )


if __name__ == "__main__":
    main()

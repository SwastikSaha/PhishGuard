#!/usr/bin/env python3
"""Classify URLs with PhishGuard and optionally explain the result with Qwen.

The ML classifier is authoritative. The LLM receives the classifier result and
transparent URL observations; it is not allowed to alter the verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import joblib

from mlp_model import URLFeatureExtractor, normalize_url

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def get_host(url: str) -> str:
    candidate = url if "://" in url else "http://" + url
    try:
        return (urlsplit(candidate).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def load_allowlist(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip().lower().removeprefix("www.").rstrip(".")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def allowlist_match(host: str, allowlist: set[str]) -> bool:
    """Tranco popularity-list match; never use this as a verdict override."""
    return bool(host) and any(
        host == domain or host.endswith("." + domain) for domain in allowlist
    )


def collect_observations(url: str, host: str, features: dict, listed: bool) -> dict:
    observations = []
    reassuring = []

    if features["is_ip_address"]:
        observations.append("The hostname is an IP address rather than a conventional domain name.")
    if "@" in url:
        observations.append("The URL contains '@', which can make the apparent destination confusing.")
    if features["suspicious_word_count"]:
        observations.append(
            f"The URL contains {features['suspicious_word_count']} occurrence(s) of configured "
            "security-sensitive terms such as login, verify, account, or password."
        )
    if features["host_hyphen_count"] >= 3:
        observations.append(f"The hostname contains {features['host_hyphen_count']} hyphens.")
    if features["subdomain_count"] >= 3:
        observations.append(f"The hostname has {features['subdomain_count']} subdomain labels.")
    if features["url_length"] >= 100:
        observations.append(f"The URL is long ({features['url_length']} characters).")
    if features["query_parameter_count"] >= 5:
        observations.append(
            f"The URL contains {features['query_parameter_count']} query parameters."
        )
    if features["has_punycode"]:
        observations.append("The hostname contains an IDN/Punycode label.")
    if features["double_slash_in_path"]:
        observations.append("The path contains a double-slash sequence.")

    # A popularity-list hit is contextual only, not proof of legitimacy.
    if listed:
        reassuring.append(
            "The hostname appears in the locally stored Tranco popularity list; "
            "that list is not a security allowlist."
        )
    else:
        reassuring.append(
            "The hostname was not found in the local Tranco list; absence is not evidence of phishing."
        )
    if "https://" in url.lower():
        reassuring.append(
            "The URL uses HTTPS, which protects the connection in transit but does not prove the site is legitimate."
        )
    if not observations:
        observations.append("None of the configured high-level URL heuristics were triggered.")

    return {
        "observed_indicators": observations,
        "contextual_observations": reassuring,
        "limitations": [
            "These are URL-string observations, not proof of malicious activity.",
            "No DNS, TLS-certificate, redirect-chain, page-content, domain-age, or live reputation lookup was performed.",
            "The ML probability is a model score, not a calibrated guarantee of real-world risk.",
            "The explanation model does not have access to hidden classifier internals or true feature attributions.",
        ],
    }


def classify(url: str, bundle: dict, allowlist: set[str]) -> dict:
    normalized = normalize_url(url)
    if not normalized:
        raise ValueError("Please provide a non-empty URL.")

    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
    phishing_probability = float(model.predict_proba([normalized])[0, 1])
    legitimate_probability = 1.0 - phishing_probability
    verdict = "PHISHING" if phishing_probability >= threshold else "LEGITIMATE"

    host = get_host(normalized)
    listed = allowlist_match(host, allowlist)

    # These are the same numeric URL features used by the current model pipeline.
    feature_values = URLFeatureExtractor().transform([normalized])[0]
    feature_names = URLFeatureExtractor.feature_names
    features = {name: float(value) for name, value in zip(feature_names, feature_values)}
    features["is_ip_address"] = bool(features["is_ip_address"])
    features["has_punycode"] = bool(features["has_punycode"])
    features["double_slash_in_path"] = bool(features["double_slash_in_path"])
    evidence = collect_observations(normalized, host, features, listed)

    return {
        "input_url": url,
        "normalized_url": normalized,
        "host": host,
        "verdict": verdict,
        "legitimate_probability": legitimate_probability,
        "phishing_probability": phishing_probability,
        "threshold": threshold,
        "tranco_listed": listed,
        "features": features,
        **evidence,
    }


def generate_qwen_explanation(
    payload: dict,
    model_id: str = MODEL_ID,
    max_new_tokens: int = 180,
    llm_device: str = "auto",
) -> str:
    """Generate an explanation from local Hugging Face Transformers weights."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen explanation dependencies are missing. Install with: "
            "python -m pip install torch transformers accelerate"
        ) from exc

    if llm_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = llm_device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"\nLoading explanation model: {model_id} ({device}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    llm.eval()

    system_prompt = (
        "You are PhishGuard's security explanation component. "
        "The supplied ML classifier verdict, probability, and threshold are authoritative. "
        "Never change, reinterpret, or override the verdict. "
        "Explain only the supplied URL observations and model output. "
        "Do not invent domain reputation, ownership, DNS records, certificate status, "
        "redirects, page contents, or model feature attributions. "
        "HTTPS is not proof of legitimacy, and Tranco popularity is not proof of safety. "
        "Clearly distinguish observed URL characteristics from the classifier's prediction. "
        "If evidence is limited, say so. Give a concise explanation and practical precautions."
    )
    user_prompt = (
        "Explain this PhishGuard result for a non-expert. State the classifier verdict "
        "without changing it, explain the strongest supplied indicators, mention uncertainty, "
        "and give practical next steps.\n\n"
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

    try:
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
    finally:
        del llm
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return explanation or "The explanation model returned an empty response."


def print_report(result: dict, explanation: str | None = None) -> None:
    print("\n" + "=" * 66)
    print("                         PHISHGUARD REPORT")
    print("=" * 66)
    print(f"URL:                    {result['input_url']}")
    print(f"Verdict:                {result['verdict']}")
    print(f"Legitimate probability: {result['legitimate_probability']:.2%}")
    print(f"Phishing probability:   {result['phishing_probability']:.2%}")
    print(f"Decision threshold:     {result['threshold']:.2%}")
    print(f"Hostname:               {result['host'] or '(unparsed)'}")

    print("\n--- URL OBSERVATIONS ---")
    for item in result["observed_indicators"]:
        print(f"[!] {item}")
    for item in result["contextual_observations"]:
        print(f"[i] {item}")

    print("\n--- EXTRACTED MODEL FEATURES ---")
    for name, value in result["features"].items():
        print(f"{name:26s}: {value}")

    if explanation:
        print("\n--- QWEN LLM EXPLANATION ---")
        print(explanation)

    print("\n--- LIMITATIONS ---")
    for item in result["limitations"]:
        print(f"- {item}")
    print("\nThe LLM explains the ML classifier output; it does not determine the verdict.")
    print("A model prediction is not a guarantee of website safety.")
    print("=" * 66)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a URL with PhishGuard and optionally explain it with Qwen."
    )
    parser.add_argument("url", nargs="?", help="URL to classify.")
    parser.add_argument("--artifacts", type=Path, default=Path("phishguard_artifacts"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--no-explanation", action="store_true", help="Skip Qwen and run ML only.")
    parser.add_argument("--llm", default=MODEL_ID, help="Hugging Face model ID for Qwen.")
    parser.add_argument("--llm-device", default="auto", help="Qwen device: auto or cpu/cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--json", action="store_true", help="Print result as JSON (no LLM explanation).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")

    url = args.url or input("Enter URL: ").strip()
    if not url:
        raise ValueError("No URL supplied.")

    model_path = args.model or (args.artifacts / "phishguard_model.joblib")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train it first with mlp_model.py."
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError(
            "Incompatible model file. Retrain using the current mlp_model.py."
        )
    allowlist_path = args.allowlist or (args.artifacts / "tranco_allowlist.txt")
    result = classify(url, bundle, load_allowlist(allowlist_path))

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    explanation = None
    if not args.no_explanation:
        try:
            explanation = generate_qwen_explanation(
                result, args.llm, args.max_new_tokens, args.llm_device
            )
        except Exception as exc:
            print(f"\n[!] Qwen explanation unavailable: {exc}", file=sys.stderr)
            print(
                "[!] Continuing with the ML verdict and deterministic observations.",
                file=sys.stderr,
            )

    print_report(result, explanation)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

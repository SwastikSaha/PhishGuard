
import json
import os
import re
import sys
import tempfile
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import tldextract


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PHISH_FILE = DATA_DIR / "phishunt_feed.json"
BENIGN_FILE = DATA_DIR / "benign_urls.json"
OUTPUT_FILE = PROCESSED_DIR / "url_features.json"


# Prevent tldextract from making network requests.
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


SENSITIVE_WORDS = [
    "login", "signin", "verify", "verification",
    "account", "update", "secure", "security",
    "password", "confirm", "banking", "wallet",
    "payment", "credential", "authenticate"
]

TRENDY_BRANDS = [
    "paypal", "amazon", "google", "facebook",
    "microsoft", "apple", "netflix", "instagram",
    "whatsapp", "linkedin", "telegram", "binance",
    "coinbase", "steam"
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc"
}


def normalize_url(url):
    url = str(url).strip()

    if not url:
        raise ValueError("Empty URL encountered.")

    if "://" not in url:
        url = "http://" + url

    return url


def get_hostname(url):
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def is_ip_address(hostname):
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_domain_features(hostname):
    extracted = EXTRACTOR(hostname)

    registered_domain = extracted.top_domain_under_public_suffix

    if not registered_domain:
        registered_domain = hostname

    return extracted, registered_domain


def contains_word(text, words):
    text = text.lower()

    for word in words:
        pattern = rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])"

        if re.search(pattern, text):
            return 1

    return 0


def extract_url_features(raw_url):
    url = normalize_url(raw_url)
    parsed = urlparse(url)

    hostname = get_hostname(url)

    extracted, registered_domain = extract_domain_features(hostname)

    path = parsed.path or ""

    # Count non-empty path segments.
    url_depth = len([
        segment for segment in path.split("/")
        if segment
    ])

    # Check for // after the scheme separator.
    remainder = url.split("://", 1)[1]
    redirection_double_slash = int("//" in remainder)

    # Check if the hostname itself is an IP address.
    ip_address = is_ip_address(hostname)

    # Count subdomains.
    count_subdomains = len(extracted.subdomain.split(".")) if (
        extracted.subdomain
    ) else 0

    # Check for hyphen in the registered domain's main label.
    domain_label = extracted.domain or hostname
    prefix_suffix_hyphen = int("-" in domain_label)

    # Check for known URL shortening services.
    url_shortener = int(
        hostname in URL_SHORTENERS
        or any(hostname.endswith("." + service) for service in URL_SHORTENERS)
    )

    # Detect HTTP/HTTPS text in the hostname.
    http_https_in_domain = int(
        "http" in hostname or "https" in hostname
    )

    # Combine URL components for word-based checks.
    searchable_url = f"{hostname} {path} {parsed.query}"

    features = {
        "domain": hostname,
        "count_subdomains": count_subdomains,
        "ip_address": ip_address,
        "at_symbol": int("@" in url),
        "url_length": len(url),
        "url_depth": url_depth,
        "redirection_double_slash": redirection_double_slash,
        "http_https_in_domain": http_https_in_domain,
        "https_scheme": int(parsed.scheme.lower() == "https"),
        "url_shortener": url_shortener,
        "prefix_suffix_hyphen": prefix_suffix_hyphen,
        "contains_sensitive_word": contains_word(
            searchable_url, SENSITIVE_WORDS
        ),
        "contains_trendy_brand": contains_word(
            searchable_url, TRENDY_BRANDS
        ),
        "uppercase_letter": int(any(char.isupper() for char in raw_url)),
        "number_of_dots": url.count(".")
    }

    return features


def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {file_path}")

    return data


def process_dataset(records, label, source):
    processed = []

    for index, record in enumerate(records):
        url = record.get("url")

        if not url:
            raise ValueError(
                f"Missing URL in {source} record {index}"
            )

        features = extract_url_features(url)

        processed.append({
            "url": url,
            "label": label,
            "source": source,
            "features": features
        })

    return processed


def atomic_json_write(data, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_file.parent,
            suffix=".tmp",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

            json.dump(data, temp_file, indent=4, ensure_ascii=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, output_file)

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def main():
    print("[Features] Loading datasets...")

    phishing_records = load_json(PHISH_FILE)
    benign_records = load_json(BENIGN_FILE)

    print(f"[Features] Phishing records: {len(phishing_records)}")
    print(f"[Features] Tranco candidates: {len(benign_records)}")

    phishing_processed = process_dataset(
        phishing_records,
        label=1,
        source="phishunt"
    )

    benign_processed = process_dataset(
        benign_records,
        label=0,
        source="tranco"
    )

    combined = phishing_processed + benign_processed

    atomic_json_write(combined, OUTPUT_FILE)

    print(f"[Features] Total processed records: {len(combined)}")
    print(f"[Features] Features per URL: 15")
    print(f"[Features] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Features] ERROR: {error}", file=sys.stderr)
        sys.exit(1)



import csv
import io
import json
import os
import random
import sys
import tempfile
import urllib.request
import zipfile
import gzip
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

PHISH_FILE = DATA_DIR / "phishunt_feed.json"
OUTPUT_FILE = DATA_DIR / "benign_urls.json"

TRANCO_METADATA_URL = "https://tranco-list.eu/api/lists/date/latest"
TIMEOUT = 60


def fetch_url(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PhishGuard/1.0"}
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP error: {response.status}")

        return response.read()


def load_phishing_count():
    if not PHISH_FILE.exists():
        raise FileNotFoundError(
            f"PhishHunt feed not found: {PHISH_FILE}"
        )

    with PHISH_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("PhishHunt feed is empty or invalid.")

    return len(records)


def get_tranco_download_url():
    metadata_bytes = fetch_url(TRANCO_METADATA_URL)
    metadata = json.loads(metadata_bytes)

    if not isinstance(metadata, dict):
        raise ValueError("Invalid Tranco metadata response.")

    download_url = metadata.get("download")

    if not download_url:
        raise ValueError("Tranco metadata contains no download URL.")

    return download_url


def extract_text(download_bytes):
    # ZIP archive
    if download_bytes.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(download_bytes)) as archive:
            names = [
                name for name in archive.namelist()
                if not name.endswith("/")
            ]

            if not names:
                raise ValueError("Tranco ZIP archive is empty.")

            with archive.open(names[0]) as file:
                return file.read().decode("utf-8-sig")

    # GZIP archive
    if download_bytes.startswith(b"\x1f\x8b"):
        return gzip.decompress(download_bytes).decode("utf-8-sig")

    # Plain text / CSV
    return download_bytes.decode("utf-8-sig")


def parse_tranco_list(text):
    domains = []
    seen = set()

    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue

        # Tranco format: rank,domain
        if len(row) >= 2:
            rank = row[0].strip()
            domain = row[1].strip().lower()
        else:
            rank = ""
            domain = row[0].strip().lower()

        if not domain or domain.lower() == "domain":
            continue

        try:
            rank = int(rank)
        except ValueError:
            continue

        if domain not in seen:
            seen.add(domain)
            domains.append({
                "domain": domain,
                "tranco_rank": rank
            })

    return domains


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
    required_count = load_phishing_count()

    print(f"[Tranco] Required candidate domains: {required_count}")
    print("[Tranco] Fetching latest list metadata...")

    download_url = get_tranco_download_url()

    print("[Tranco] Downloading list...")
    download_bytes = fetch_url(download_url)

    text = extract_text(download_bytes)
    domains = parse_tranco_list(text)

    if len(domains) < required_count:
        raise RuntimeError(
            f"Tranco provided only {len(domains)} unique domains; "
            f"{required_count} are required."
        )

    selected = random.SystemRandom().sample(
        domains,
        required_count
    )

    output_records = []

    for item in selected:
        domain = item["domain"]

        output_records.append({
            "url": f"https://{domain}/",
            "domain": domain,
            "tranco_rank": item["tranco_rank"]
        })

    atomic_json_write(output_records, OUTPUT_FILE)

    print(f"[Tranco] Selected {len(output_records)} domains.")
    print(f"[Tranco] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Tranco] ERROR: {error}", file=sys.stderr)
        sys.exit(1)


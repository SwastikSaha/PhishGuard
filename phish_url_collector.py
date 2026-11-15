
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "phishunt_feed.json"

FEED_URL = "https://phishunt.io/api/v1/domains?format=json"
TIMEOUT = 30


def download_feed():
    request = urllib.request.Request(
        FEED_URL,
        headers={"User-Agent": "PhishGuard/1.0"}
    )

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 200:
            raise RuntimeError(
                f"PhishHunt returned HTTP {response.status}"
            )

        payload = response.read()

    return json.loads(payload)



def validate_feed(data):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the PhishHunt API.")

    if "results" not in data:
        raise ValueError(
            "PhishHunt response does not contain a 'results' field."
        )

    records = data["results"]

    if not isinstance(records, list):
        raise ValueError("PhishHunt 'results' must be a JSON array.")

    if not records:
        raise ValueError("PhishHunt returned an empty feed.")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record at index {index}.")

        if not record.get("url") or not record.get("domain"):
            raise ValueError(
                f"Record {index} is missing its URL or domain."
            )

    print(
        f"[PhishHunt] API reports {data.get('total', len(records))} "
        f"total records; received {len(records)} records."
    )

    return records



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
    print("[PhishHunt] Downloading latest feed...")

    feed = download_feed()
    feed = validate_feed(feed)

    atomic_json_write(feed, OUTPUT_FILE)

    print(f"[PhishHunt] Downloaded {len(feed)} records.")
    print(f"[PhishHunt] Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[PhishHunt] ERROR: {error}", file=sys.stderr)
        sys.exit(1)


#!/usr/bin/env python3
"""Build a domain-separated URL dataset for PhishGuard.

Labels: 1 = phishing, 0 = legitimate.
The script uses the active Phishing.Database feed and the Tranco top-sites list.
"""
from __future__ import annotations

import argparse
import csv
import io
import ipaddress
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PHISHING_FEED = (
    "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
    "master/phishing-links-ACTIVE.txt"
)
TRANCO_ZIP = "https://tranco-list.eu/top-1m.csv.zip"
USER_AGENT = "PhishGuard-DatasetBuilder/2.0"


def download_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def normalize_url(value: object) -> str:
    """Canonicalise scheme/www without rejecting unusual but usable URLs.

    Hostnames are lowercased; path/query case is preserved because URL paths
    can be case-sensitive. Invalid parser edge cases fall back to the raw text.
    """
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return ""
    candidate = raw if "://" in raw else "http://" + raw.lstrip("/")
    try:
        parsed = urlsplit(candidate)
        host = parsed.hostname or ""
        # Accessing .port can raise ValueError on malformed ports.
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
    netloc = host + (f":{port}" if port is not None else "")
    path = parsed.path or "/"
    result = netloc + path
    if parsed.query:
        result += "?" + parsed.query
    if parsed.fragment:
        result += "#" + parsed.fragment
    return result


def extract_host(url: str) -> str:
    if not url:
        return ""
    candidate = url if "://" in url else "http://" + url
    try:
        return (urlsplit(candidate).hostname or "").rstrip(".").lower()
    except ValueError:
        # Best-effort fallback for malformed bracket/authority edge cases.
        authority = candidate.split("/", 3)[2] if candidate.count("/") >= 2 else ""
        authority = authority.rsplit("@", 1)[-1].split(":", 1)[0]
        return authority.strip("[]").rstrip(".").lower()


_MULTI_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.in", "firm.in", "net.in", "org.in", "gen.in", "ind.in",
    "co.jp", "ne.jp", "or.jp", "com.br", "com.cn", "com.sg", "co.nz",
}


def registrable_domain(host: str) -> str:
    """Return a practical registrable-domain grouping key.

    Uses tldextract when installed (with network updates disabled); otherwise
    uses a conservative common-multi-suffix fallback.
    """
    host = (host or "").strip(".").lower()
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    try:
        import tldextract  # optional, recommended
        extracted = tldextract.TLDExtract(suffix_list_urls=())(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
    except Exception:
        pass
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def parse_phishing_feed(content: bytes, limit: int) -> list[dict]:
    text = content.decode("utf-8", errors="replace")
    rows, seen = [], set()
    for line in text.splitlines():
        item = line.strip().strip('"').strip("'")
        if not item or item.startswith(("#", ";")):
            continue
        # Ignore obvious feed metadata/header rows.
        if item.lower() in {"url", "phishing_url", "domain"}:
            continue
        normalized = normalize_url(item)
        host = extract_host(normalized)
        if not normalized or not host:
            continue
        domain = registrable_domain(host)
        if not domain or normalized in seen:
            continue
        seen.add(normalized)
        rows.append({"url": normalized, "label": 1, "domain": domain})
        if len(rows) >= limit:
            break
    return rows


def parse_tranco_zip(content: bytes, limit: int) -> list[dict]:
    rows, seen = [], set()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("Tranco archive did not contain a CSV file.")
        with archive.open(csv_names[0]) as stream:
            reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8", errors="replace"))
            for record in reader:
                if not record:
                    continue
                host = record[1].strip().lower() if len(record) > 1 else record[0].strip().lower()
                host = host.removeprefix("www.").rstrip(".")
                if not host or host in seen:
                    continue
                domain = registrable_domain(host)
                if not domain:
                    continue
                seen.add(host)
                rows.append({"url": normalize_url(host), "label": 0, "domain": domain})
                if len(rows) >= limit:
                    break
    return rows


def make_group_splits(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Create approximately 70/15/15 splits with no domain crossing."""
    groups = frame["domain"].astype(str).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, hold_idx = next(first.split(frame, frame["label"], groups))
    train = frame.iloc[train_idx].copy()
    hold = frame.iloc[hold_idx].copy()

    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_rel, test_rel = next(second.split(hold, hold["label"], hold["domain"]))
    validation = hold.iloc[val_rel].copy()
    test = hold.iloc[test_rel].copy()

    splits = {"train": train, "validation": validation, "test": test}
    domain_sets = {name: set(part["domain"]) for name, part in splits.items()}
    overlaps = {
        "train_validation": len(domain_sets["train"] & domain_sets["validation"]),
        "train_test": len(domain_sets["train"] & domain_sets["test"]),
        "validation_test": len(domain_sets["validation"] & domain_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Domain leakage detected: {overlaps}")
    for name, part in splits.items():
        if part["label"].nunique() < 2:
            raise RuntimeError(f"{name} split contains only one class; obtain more source domains.")
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PhishGuard URL dataset.")
    parser.add_argument("--phishing-count", type=int, default=100000)
    parser.add_argument("--benign-count", type=int, default=100000)
    parser.add_argument("--artifacts", type=Path, default=Path("phishguard_artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feed-url", default=PHISHING_FEED)
    parser.add_argument("--tranco-url", default=TRANCO_ZIP)
    args = parser.parse_args()

    if args.phishing_count < 2 or args.benign_count < 2:
        parser.error("Both requested class counts must be at least 2.")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    print("[1/4] Downloading active phishing feed...")
    phishing = parse_phishing_feed(download_bytes(args.feed_url), args.phishing_count)
    if len(phishing) < 2:
        raise RuntimeError("The phishing feed returned fewer than two usable URLs.")

    print(f"      Collected {len(phishing):,} unique phishing URLs.")
    phishing_domains = {row["domain"] for row in phishing}

    print("[2/4] Downloading Tranco top-sites list...")
    benign_candidates = parse_tranco_zip(download_bytes(args.tranco_url), args.benign_count * 3)
    benign = [row for row in benign_candidates if row["domain"] not in phishing_domains]
    benign = benign[:args.benign_count]
    if len(benign) < 2:
        raise RuntimeError("Not enough conflict-free Tranco domains were available.")

    # Use the same number of samples per class where possible.
    count = min(len(phishing), len(benign))
    phishing, benign = phishing[:count], benign[:count]
    frame = pd.DataFrame(phishing + benign)
    frame = frame.drop_duplicates(subset=["url"]).reset_index(drop=True)
    frame["label"] = frame["label"].astype("int8")
    frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print("[3/4] Creating domain-separated train/validation/test splits...")
    splits = make_group_splits(frame, args.seed)
    frame.to_csv(args.artifacts / "url_dataset.csv", index=False)
    for name, part in splits.items():
        part.to_csv(args.artifacts / f"{name}.csv", index=False)

    allowlist = sorted({row["domain"] for row in benign})
    (args.artifacts / "tranco_allowlist.txt").write_text(
        "\n".join(allowlist) + "\n", encoding="utf-8"
    )

    split_report = {}
    for name, part in splits.items():
        split_report[name] = {
            "rows": int(len(part)),
            "class_counts": {str(k): int(v) for k, v in part["label"].value_counts().items()},
            "unique_domains": int(part["domain"].nunique()),
        }
    domain_sets = {name: set(part["domain"]) for name, part in splits.items()}
    report = {
        "source_phishing_feed": args.feed_url,
        "source_benign_list": args.tranco_url,
        "rows": int(len(frame)),
        "class_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "splits": split_report,
        "domain_overlap": {
            "train_validation": len(domain_sets["train"] & domain_sets["validation"]),
            "train_test": len(domain_sets["train"] & domain_sets["test"]),
            "validation_test": len(domain_sets["validation"] & domain_sets["test"]),
        },
        "normalisation": "scheme removed; one leading www. removed; host lowercased; path/query case preserved",
        "seed": args.seed,
    }
    (args.artifacts / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("[4/4] Dataset ready.")
    print(f"      Rows: {len(frame):,} | classes: {report['class_counts']}")
    print(f"      Splits: { {k: v['rows'] for k, v in split_report.items()} }")
    print(f"      Domain overlap: {report['domain_overlap']}")
    print(f"      Output: {args.artifacts.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

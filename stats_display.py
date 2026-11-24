#!/usr/bin/env python3
"""Display PhishGuard dataset and model evaluation reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Show PhishGuard statistics.")
    parser.add_argument("--artifacts", type=Path, default=Path("phishguard_artifacts"))
    parser.add_argument("--plot", action="store_true", help="Display test confusion matrix.")
    args = parser.parse_args()

    dataset = read_json(args.artifacts / "dataset_report.json")
    metrics = read_json(args.artifacts / "model_metrics.json")

    if dataset is None and metrics is None:
        raise FileNotFoundError(
            f"No reports found in {args.artifacts}. Run dataset_manager.py and model.py first."
        )

    if dataset:
        print("\n=== DATASET ===")
        print(f"Rows: {dataset.get('rows', 'n/a'):,}")
        print(f"Class counts: {dataset.get('class_counts', {})}")
        print("Splits:")
        for name, details in dataset.get("splits", {}).items():
            print(
                f"  {name:12s} rows={details.get('rows')} "
                f"domains={details.get('unique_domains')} "
                f"classes={details.get('class_counts')}"
            )
        print(f"Domain overlap: {dataset.get('domain_overlap', {})}")

    if metrics:
        print("\n=== MODEL ===")
        print(f"Model: {metrics.get('model')}")
        print(f"Threshold policy: {metrics.get('threshold_selection')}")
        for split in ("validation", "test"):
            values = metrics.get(split)
            if not values:
                continue
            print(f"\n{split.upper()}")
            for key in ("threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"):
                value = values.get(key)
                if isinstance(value, float):
                    print(f"  {key:12s}: {value:.6f}")
                else:
                    print(f"  {key:12s}: {value}")
            print(
                f"  confusion   : TN={values.get('tn')} FP={values.get('fp')} "
                f"FN={values.get('fn')} TP={values.get('tp')}"
            )

    if args.plot:
        if not metrics or "test" not in metrics:
            raise RuntimeError("Model metrics are not available; train the model first.")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        test = metrics["test"]
        matrix = [[test["tn"], test["fp"]], [test["fn"], test["tp"]]]
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["Legitimate", "Phishing"]
        )
        display.plot(values_format="d")
        plt.title("PhishGuard — Held-out Test Confusion Matrix")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

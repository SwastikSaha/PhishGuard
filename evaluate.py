
import hashlib
import math
import json
import shutil
import os

from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from optimize import ARTIFACTS, load_dataset


CANDIDATE_DIR = ARTIFACTS / "candidate"
CURRENT_DIR = ARTIFACTS / "current"
PREVIOUS_DIR = ARTIFACTS / "previous"

MODEL_NAMES = [
    "catboost",
    "lightgbm",
    "xgboost",
]


def dataset_fingerprint(records):
    serialized = json.dumps(
        records,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def load_test_indices(records):
    split_file = ARTIFACTS / "splits.json"

    with split_file.open("r") as file:
        splits = json.load(file)

    current_fingerprint = dataset_fingerprint(records)

    if splits["fingerprint"] != current_fingerprint:
        raise RuntimeError(
            "Dataset changed after training. "
            "Candidate models cannot be evaluated on this dataset."
        )

    return np.array(splits["test_idx"])


def promote_candidate():
    if not (CANDIDATE_DIR / "evaluation.json").exists():
        raise RuntimeError("Candidate evaluation is missing.")

    if PREVIOUS_DIR.exists():
        shutil.rmtree(PREVIOUS_DIR)

    if CURRENT_DIR.exists():
        CURRENT_DIR.rename(PREVIOUS_DIR)

    try:
        CANDIDATE_DIR.rename(CURRENT_DIR)
    except Exception:
        if PREVIOUS_DIR.exists() and not CURRENT_DIR.exists():
            PREVIOUS_DIR.rename(CURRENT_DIR)
        raise

    print("[Evaluation] Candidate promoted to current.")
    print("[Evaluation] Previous model set retained for rollback.")


def main():
    print("[Evaluation] Loading latest dataset...")

    X, y, groups, records = load_dataset()

    test_idx = load_test_indices(records)

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    base_predictions = []

    print("\n[Evaluation] Loading candidate base models...")

    for model_name in MODEL_NAMES:
        model_path = CANDIDATE_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Candidate model missing: {model_path}"
            )

        model = joblib.load(model_path)

        probabilities = model.predict_proba(X_test)[:, 1]

        base_predictions.append(probabilities)

        print(f"[Evaluation] Loaded {model_name}.")

    X_meta_test = np.column_stack(base_predictions)

    meta_path = CANDIDATE_DIR / "meta_learner.joblib"

    if not meta_path.exists():
        raise FileNotFoundError("Candidate meta-learner missing.")

    meta_model = joblib.load(meta_path)

    probabilities = meta_model.predict_proba(X_meta_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "log_loss": float(
            log_loss(y_test, probabilities, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Tranco candidate", "PhishHunt"],
            output_dict=True,
            zero_division=0,
        ),
    }

    print("\n" + "=" * 55)
    print("PHISHGUARD — STACKING ENSEMBLE EVALUATION")
    print("=" * 55)

    for key, value in results.items():
        if key not in [
            "confusion_matrix",
            "classification_report",
        ]:
            print(f"{key}: {value}")

    print("\nConfusion Matrix:")
    print(np.array(results["confusion_matrix"]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Tranco candidate", "PhishHunt"],
            zero_division=0,
        )
    )

    output_file = CANDIDATE_DIR / "evaluation.json"

    with output_file.open("w") as file:
        json.dump(results, file, indent=4)

    print(f"\n[Evaluation] Results saved to {output_file}")

    minimum_roc_auc = float(os.getenv("PHISHGUARD_MIN_ROC_AUC", "0.80"))
    minimum_recall = float(os.getenv("PHISHGUARD_MIN_RECALL", "0.70"))

    print(
        f"\\n[Evaluation] Promotion gates: ROC-AUC >= {minimum_roc_auc:.2f}, "
        f"recall >= {minimum_recall:.2f}"
    )

    if (
        not math.isfinite(results["roc_auc"])
        or not math.isfinite(results["recall"])
        or results["roc_auc"] < minimum_roc_auc
        or results["recall"] < minimum_recall
    ):
        print(
            "[Evaluation] Candidate did not meet promotion gates. "
            "Current model set has been left unchanged."
        )
        return

    promote_candidate()


if __name__ == "__main__":
    main()



import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

from optimize import (
    ARTIFACTS,
    RANDOM_STATE,
    build_model,
    load_dataset,
)


CANDIDATE_DIR = ARTIFACTS / "candidate"

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


def create_splits(y, groups):
    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    remaining_idx, test_idx = next(
        outer_splitter.split(
            np.zeros(len(y)),
            y,
            groups,
        )
    )

    inner_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )

    train_relative, val_relative = next(
        inner_splitter.split(
            np.zeros(len(remaining_idx)),
            y[remaining_idx],
            groups[remaining_idx],
        )
    )

    train_idx = remaining_idx[train_relative]
    val_idx = remaining_idx[val_relative]

    return train_idx, val_idx, test_idx


def save_splits(records, train_idx, val_idx, test_idx):
    split_data = {
        "fingerprint": dataset_fingerprint(records),
        "train_idx": train_idx.tolist(),
        "validation_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

    with (ARTIFACTS / "splits.json").open("w") as file:
        json.dump(split_data, file, indent=4)


def main():
    print("[Training] Loading latest feature dataset...")

    X, y, groups, records = load_dataset()

    if len(np.unique(y)) != 2:
        raise ValueError("Both classes are required for training.")

    if len(np.unique(groups)) < 5:
        raise ValueError(
            "At least five distinct domain groups are required."
        )

    train_idx, val_idx, test_idx = create_splits(y, groups)

    save_splits(records, train_idx, val_idx, test_idx)

    print(f"[Training] Total records: {len(y)}")
    print(f"[Training] Training: {len(train_idx)}")
    print(f"[Training] Validation: {len(val_idx)}")
    print(f"[Training] Test: {len(test_idx)}")

    # Create a clean candidate directory.
    if CANDIDATE_DIR.exists():
        shutil.rmtree(CANDIDATE_DIR)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    with (ARTIFACTS / "best_params.json").open("r") as file:
        optimization_results = json.load(file)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    oof_predictions = np.zeros(
        (len(train_idx), len(MODEL_NAMES)),
        dtype=float,
    )

    # -------------------------------------------------
    # Generate out-of-fold predictions
    # -------------------------------------------------

    print("\n[Training] Generating OOF predictions...")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold, (fold_train, fold_valid) in enumerate(
        splitter.split(X_train, y_train, groups_train),
        start=1,
    ):
        print(f"\n[Training] Fold {fold}/5")

        for model_index, model_name in enumerate(MODEL_NAMES):
            params = optimization_results[model_name]["best_params"]

            model = build_model(model_name, params)

            model.fit(
                X_train.iloc[fold_train],
                y_train[fold_train],
            )

            probabilities = model.predict_proba(
                X_train.iloc[fold_valid]
            )[:, 1]

            oof_predictions[fold_valid, model_index] = probabilities

            print(f"[Training] {model_name} fold completed.")

    if not np.isfinite(oof_predictions).all():
        raise RuntimeError("Invalid OOF predictions generated.")

    np.savez_compressed(
        CANDIDATE_DIR / "oof_predictions.npz",
        predictions=oof_predictions,
        labels=y_train,
        model_names=np.array(MODEL_NAMES),
    )

    # -------------------------------------------------
    # Train final base models from scratch
    # -------------------------------------------------

    print("\n[Training] Fitting final base models...")

    development_idx = np.concatenate([
        train_idx,
        val_idx,
    ])

    X_development = X.iloc[development_idx]
    y_development = y[development_idx]

    for model_name in MODEL_NAMES:
        print(f"\n[Training] Training {model_name}...")

        params = optimization_results[model_name]["best_params"]

        model = build_model(model_name, params)

        model.fit(X_development, y_development)

        model_path = CANDIDATE_DIR / f"{model_name}.joblib"

        import joblib
        joblib.dump(model, model_path)

        print(f"[Training] Saved {model_path}")

    metadata = {
        "dataset_fingerprint": dataset_fingerprint(records),
        "model_names": MODEL_NAMES,
        "training_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "development_samples": len(development_idx),
        "test_samples": len(test_idx),
        "random_state": RANDOM_STATE,
    }

    with (CANDIDATE_DIR / "training_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=4)

    print("\n[Training] All base models trained successfully.")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "url_length",
    "url_entropy",
    "slash_count",
    "query_length",
    "domain_length",
    "domain_entropy",
    "sub_domain",
    "digit_count",
    "special_char_count",
    "hyphen_count",
    "https_flag",
    "ip_flag",
    "at_flag",
    "keyword_flag",
]
CATEGORICAL_FEATURES = ["tld"]
TARGET = "label"
URL_COLUMN = "url"
DOMAIN_COLUMN = "domain"


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def grouped_split(X, y, groups, seed=42):

    first = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=seed
    )
    train_val_idx, test_idx = next(first.split(X, y, groups))

    second = GroupShuffleSplit(
        n_splits=1, test_size=(0.15 / 0.85), random_state=seed
    )
    train_rel, val_rel = next(
        second.split(
            X.iloc[train_val_idx],
            y.iloc[train_val_idx],
            groups.iloc[train_val_idx],
        )
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Prepare PhishGuard URL dataset.")
    parser.add_argument("--csv", required=True, help="Path to the source CSV.")
    parser.add_argument("--output", default="phishguard_artifacts",
                        help="Directory for processed splits and preprocessing.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path}")
    print(f"Original shape: {df.shape}")

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
        TARGET, URL_COLUMN, DOMAIN_COLUMN
    ]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    # Remove duplicate URL records and invalid target rows.
    df = df.drop_duplicates(subset=[URL_COLUMN]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype("int64")

    # Convert numeric columns; invalid values become NaN and are imputed later.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df["tld"] = df["tld"].fillna(np.nan).astype("object")

    # Missing domains get a URL-specific group so unrelated missing-domain
    # records are not all forced into one giant group.
    groups = df[DOMAIN_COLUMN].astype("object").copy()
    missing_domain = groups.isna() | (groups.astype(str).str.strip() == "")
    groups.loc[missing_domain] = (
        "__missing_domain__" + df.loc[missing_domain, URL_COLUMN].astype(str)
    )
    groups = groups.astype(str)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    train_idx, val_idx, test_idx = grouped_split(
        X, y, groups, seed=args.seed
    )

    X_train, X_val, X_test = (
        X.iloc[train_idx].copy(),
        X.iloc[val_idx].copy(),
        X.iloc[test_idx].copy(),
    )
    y_train, y_val, y_test = (
        y.iloc[train_idx].copy(),
        y.iloc[val_idx].copy(),
        y.iloc[test_idx].copy(),
    )
    g_train, g_val, g_test = (
        groups.iloc[train_idx],
        groups.iloc[val_idx],
        groups.iloc[test_idx],
    )

    # Fit only on training data to avoid preprocessing leakage.
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    np.savez_compressed(
        output_dir / "train.npz",
        X=X_train_processed, y=y_train.to_numpy(dtype=np.int64)
    )
    np.savez_compressed(
        output_dir / "validation.npz",
        X=X_val_processed, y=y_val.to_numpy(dtype=np.int64)
    )
    np.savez_compressed(
        output_dir / "test.npz",
        X=X_test_processed, y=y_test.to_numpy(dtype=np.int64)
    )
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    feature_names = preprocessor.get_feature_names_out().tolist()
    domain_overlap = {
        "train_validation": len(set(g_train) & set(g_val)),
        "train_test": len(set(g_train) & set(g_test)),
        "validation_test": len(set(g_val) & set(g_test)),
    }

    report = {
        "source_csv": str(csv_path),
        "rows_after_cleaning": int(len(df)),
        "source_columns": df.columns.tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "processed_feature_count": int(X_train_processed.shape[1]),
        "processed_feature_names": feature_names,
        "split_shapes": {
            "train": list(X_train_processed.shape),
            "validation": list(X_val_processed.shape),
            "test": list(X_test_processed.shape),
        },
        "class_counts": {
            "train": y_train.value_counts().sort_index().to_dict(),
            "validation": y_val.value_counts().sort_index().to_dict(),
            "test": y_test.value_counts().sort_index().to_dict(),
        },
        "domain_overlap": domain_overlap,
        "seed": args.seed,
    }
    # JSON keys/values need to be plain Python types.
    report["class_counts"] = {
        split: {str(k): int(v) for k, v in counts.items()}
        for split, counts in report["class_counts"].items()
    }
    with open(output_dir / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nProcessed split shapes:")
    print("Train:     ", X_train_processed.shape)
    print("Validation:", X_val_processed.shape)
    print("Test:      ", X_test_processed.shape)
    print("\nDomain overlap:", domain_overlap)
    print("\nClass counts:")
    for split, counts in report["class_counts"].items():
        print(f"{split:10s}: {counts}")
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

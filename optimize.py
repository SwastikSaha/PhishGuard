
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tldextract

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data/processed/url_features.json"
ARTIFACTS = ROOT / "ml/artifacts"

ARTIFACTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 20

FEATURE_COLUMNS = [
    "count_subdomains",
    "ip_address",
    "at_symbol",
    "url_length",
    "url_depth",
    "redirection_double_slash",
    "http_https_in_domain",
    "https_scheme",
    "url_shortener",
    "prefix_suffix_hyphen",
    "contains_sensitive_word",
    "contains_trendy_brand",
    "uppercase_letter",
    "number_of_dots",
]

EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def load_dataset():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("Feature dataset is empty or invalid.")

    X = pd.DataFrame([
        record["features"] for record in records
    ])

    missing = set(FEATURE_COLUMNS) - set(X.columns)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    X = X[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        raise ValueError(
            "Missing or non-numeric feature values detected. "
            "Clean the feature dataset before training."
        )

    y = np.array([record["label"] for record in records])

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")

    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be present.")

    domains = []

    for record in records:
        hostname = record["features"]["domain"].lower()
        extracted = EXTRACTOR(hostname)

        group = extracted.top_domain_under_public_suffix or hostname
        domains.append(group)

    return X, y, np.array(domains), records


def create_splits(y, groups):
    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    remaining_idx, test_idx = next(
        outer.split(np.zeros(len(y)), y, groups)
    )

    inner = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE + 1
    )

    train_relative, val_relative = next(
        inner.split(
            np.zeros(len(remaining_idx)),
            y[remaining_idx],
            groups[remaining_idx]
        )
    )

    train_idx = remaining_idx[train_relative]
    val_idx = remaining_idx[val_relative]

    return train_idx, val_idx, test_idx


def save_splits(train_idx, val_idx, test_idx, records):
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    split_data = {
        "fingerprint": fingerprint,
        "train_idx": train_idx.tolist(),
        "validation_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

    with (ARTIFACTS / "splits.json").open("w") as file:
        json.dump(split_data, file, indent=4)


def load_splits(records):
    import hashlib

    path = ARTIFACTS / "splits.json"

    if not path.exists():
        raise FileNotFoundError(
            "Splits not found. Run optimize.py first."
        )

    with path.open("r") as file:
        splits = json.load(file)

    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if splits["fingerprint"] != fingerprint:
        raise RuntimeError(
            "Dataset changed since optimization. "
            "Run optimize.py again before training."
        )

    return (
        np.array(splits["train_idx"]),
        np.array(splits["validation_idx"]),
        np.array(splits["test_idx"]),
    )


def build_model(name, params):
    if name == "catboost":
        return CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    if name == "lightgbm":
        return LGBMClassifier(
            **params,
            objective="binary",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )

    if name == "xgboost":
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {name}")


def objective_factory(model_name, X_train, y_train, X_val, y_val):
    def objective(trial):
        if model_name == "catboost":
            params = {
                "iterations": trial.suggest_int(
                    "iterations", 400, 1200
                ),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1.0, 10.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 0.0, 2.0
                ),
                "border_count": trial.suggest_int(
                    "border_count", 32, 128
                ),
            }

        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 400, 1200
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", 15, 63
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 4, 12
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 10, 60
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.7, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.7, 1.0
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-4, 10.0, log=True
                ),
                "subsample_freq": 1,
            }

        else:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 400, 1200
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1, 10
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.7, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.7, 1.0
                ),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-4, 10.0, log=True
                ),
            }

        model = build_model(model_name, params)

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        return roc_auc_score(y_val, probabilities)

    return objective


def main():
    print("[Optimization] Loading dataset...")

    X, y, groups, records = load_dataset()

    train_idx, val_idx, test_idx = create_splits(y, groups)
    save_splits(train_idx, val_idx, test_idx, records)

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y[val_idx]

    print(f"[Optimization] Total samples: {len(y)}")
    print(f"[Optimization] Training: {len(train_idx)}")
    print(f"[Optimization] Validation: {len(val_idx)}")
    print(f"[Optimization] Test: {len(test_idx)}")

    all_results = {}

    for model_name in ["catboost", "lightgbm", "xgboost"]:
        print(f"\n[Optimization] Optimizing {model_name}...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE
            )
        )

        study.optimize(
            objective_factory(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val
            ),
            n_trials=N_TRIALS
        )

        all_results[model_name] = {
            "best_auc": study.best_value,
            "best_params": study.best_params,
        }

        print(f"[Optimization] Best ROC-AUC: {study.best_value:.5f}")
        print(f"[Optimization] Best parameters: {study.best_params}")

    output = ARTIFACTS / "best_params.json"

    with output.open("w") as file:
        json.dump(all_results, file, indent=4)

    print(f"\n[Optimization] Results saved to {output}")


if __name__ == "__main__":
    main()

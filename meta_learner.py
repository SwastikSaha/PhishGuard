
import joblib
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CANDIDATE_DIR = ROOT / "ml" / "artifacts" / "candidate"


def main():
    predictions_file = CANDIDATE_DIR / "oof_predictions.npz"

    if not predictions_file.exists():
        raise FileNotFoundError(
            "OOF predictions missing. Run train.py first."
        )

    print("[Meta] Loading OOF predictions...")

    data = np.load(predictions_file)

    X_meta = data["predictions"]
    y_meta = data["labels"]

    if X_meta.shape[1] != 3:
        raise ValueError("Expected predictions from three base models.")

    if not np.isfinite(X_meta).all():
        raise ValueError("Invalid values in OOF predictions.")

    meta_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        )),
    ])

    print("[Meta] Training Logistic Regression...")

    meta_model.fit(X_meta, y_meta)

    model_path = CANDIDATE_DIR / "meta_learner.joblib"

    joblib.dump(meta_model, model_path)

    print(f"[Meta] Saved candidate meta-learner: {model_path}")


if __name__ == "__main__":
    main()


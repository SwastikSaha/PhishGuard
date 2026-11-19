"""
Expected input artifacts (created by dataset_manager.py):
    phishguard_artifacts/train.npz
    phishguard_artifacts/validation.npz
    phishguard_artifacts/test.npz
    phishguard_artifacts/preprocessor.joblib
Outputs:
    phishguard_mlp.pth
    training_history.csv
    test_metrics.json
    test_confusion_matrix.csv
    model_config.json
Run:
    python mlp_model.py
    python mlp_model.py --artifacts phishguard_artifacts --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return device


class URLClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int] = (256, 128),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if len(hidden_sizes) != 2 or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive integers.")
        if num_classes != 2:
            raise ValueError("This implementation is configured for 2 classes.")
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def load_split(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset artifact: {path}\n"
            "Run dataset_manager.py first."
        )
    with np.load(path) as data:
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
    if X.ndim != 2:
        raise ValueError(f"{path.name}: X must be a 2D matrix; got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"{path.name}: y must be a 1D vector; got {y.shape}.")
    if len(X) != len(y):
        raise ValueError(f"{path.name}: X and y have different row counts.")
    if not np.isfinite(X).all():
        raise ValueError(f"{path.name}: X contains NaN or infinite values.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path.name}: labels must be 0 or 1.")
    return torch.from_numpy(X), torch.from_numpy(y)


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)

            labels_all.append(labels.cpu().numpy())
            predictions_all.append(predictions.cpu().numpy())
            probabilities_all.append(probabilities[:, 1].cpu().numpy())

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions_all)
    y_phishing_probability = np.concatenate(probabilities_all)

    metrics = {
        "loss": float(total_loss / len(loader.dataset)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            y_true, y_phishing_probability
        )),
    }

    return metrics, y_true, y_pred, y_phishing_probability


def train(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)

    artifacts_dir = Path(args.artifacts).expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading data from: {artifacts_dir}")

    X_train, y_train = load_split(artifacts_dir / "train.npz")
    X_val, y_val = load_split(artifacts_dir / "validation.npz")
    X_test, y_test = load_split(artifacts_dir / "test.npz")

    input_size = X_train.shape[1]

    if X_val.shape[1] != input_size or X_test.shape[1] != input_size:
        raise ValueError("Train/validation/test feature dimensions do not match.")

    for split_name, labels in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        counts = torch.bincount(labels, minlength=2).tolist()
        print(
            f"{split_name:12s}: {len(labels):,} samples | "
            f"legitimate={counts[0]:,}, phishing={counts[1]:,}"
        )

    train_loader = make_loader(
        X_train, y_train, args.batch_size, True, device
    )
    val_loader = make_loader(
        X_val, y_val, args.batch_size, False, device
    )
    test_loader = make_loader(
        X_test, y_test, args.batch_size, False, device
    )

    model = URLClassifier(
        input_size=input_size,
        hidden_sizes=(args.hidden1, args.hidden2),
        num_classes=2,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nModel:")
    print(model)
    print(f"Trainable parameters: {parameter_count:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    print("\n========== TRAINING ==========")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["roc_auc"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(
                best_state,
                artifacts_dir / "phishguard_mlp.pth",
            )
            status = "Best model saved"
        else:
            patience_counter += 1
            status = f"No improvement ({patience_counter}/{args.patience})"

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"{status}"
        )

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best checkpoint.")

    pd.DataFrame(history).to_csv(
        artifacts_dir / "training_history.csv",
        index=False,
    )

    model.load_state_dict(best_state)
    model.eval()

    print(f"\nBest checkpoint: epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    test_metrics, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, device
    )

    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["test_samples"] = int(len(y_true))
    test_metrics["trainable_parameters"] = int(parameter_count)

    with open(artifacts_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["actual_legitimate", "actual_phishing"],
        columns=["predicted_legitimate", "predicted_phishing"],
    ).to_csv(artifacts_dir / "test_confusion_matrix.csv")

    print("\n========== FINAL TEST RESULTS ==========")
    for name, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{name:24s}: {value:.4f}")
        else:
            print(f"{name:24s}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(cm)

    config = {
        "input_size": int(input_size),
        "hidden_sizes": [int(args.hidden1), int(args.hidden2)],
        "num_classes": 2,
        "class_mapping": {
            "0": "Legitimate",
            "1": "Phishing",
        },
        "best_epoch": int(best_epoch),
        "trainable_parameters": int(parameter_count),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }
    with open(artifacts_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll model outputs saved in: {artifacts_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate"
    )
    parser.add_argument("--artifacts", default="phishguard_artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or another torch device.")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience and batch-size must be positive.")
    if args.hidden1 < 1 or args.hidden2 < 1:
        parser.error("Hidden layer sizes must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid learning rate or weight decay.")

    return args


if __name__ == "__main__":
    train(parse_args())

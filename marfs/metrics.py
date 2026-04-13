"""Metrics and evaluation utilities for MARFS."""

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    episode: int
    reward: float
    accuracy: float
    n_selected: int
    compression: float
    policy_loss: float
    value_loss: float
    entropy: float
    steps: int
    time: float


def evaluate_final_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Evaluate a final feature selection on held-out test data."""
    selected = np.where(mask > 0)[0]
    if len(selected) == 0:
        return {"accuracy": 0.0, "rf_accuracy": 0.0, "n_selected": 0}

    X_tr = X_train[:, selected]
    X_te = X_test[:, selected]

    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(X_tr, y_train)
    ridge_acc = accuracy_score(y_test, ridge.predict(X_te))

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_te))

    return {
        "accuracy": ridge_acc,
        "rf_accuracy": rf_acc,
        "n_selected": len(selected),
    }


def compute_jaccard_similarity(masks: list[np.ndarray]) -> float:
    """Compute mean pairwise Jaccard similarity across feature masks."""
    if len(masks) < 2:
        return 1.0

    sims = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            set_i = set(np.where(masks[i] > 0)[0])
            set_j = set(np.where(masks[j] > 0)[0])
            if len(set_i) == 0 and len(set_j) == 0:
                sims.append(1.0)
            elif len(set_i | set_j) == 0:
                sims.append(0.0)
            else:
                sims.append(len(set_i & set_j) / len(set_i | set_j))

    return float(np.mean(sims))


def compute_all_metrics(results_list: list[dict]) -> dict:
    """Aggregate metrics across multiple runs for reporting."""
    if not results_list:
        return {}

    keys = [
        "downstream_accuracy", "downstream_rf_accuracy",
        "compression", "convergence_step", "stability",
        "total_time", "n_selected", "peak_memory_mb",
    ]

    agg = {}
    for key in keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    masks = [np.array(r["best_mask"]) for r in results_list if "best_mask" in r]
    agg["cross_run_stability"] = compute_jaccard_similarity(masks)

    return agg

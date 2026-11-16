# utils_ml.py

import random

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def safe_auc_score(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        metrics["sensitivity"] = (
            float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        )
        metrics["specificity"] = (
            float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        )
    else:
        metrics["sensitivity"] = np.nan
        metrics["specificity"] = np.nan
    metrics["auc"] = (
        safe_auc_score(y_true, np.asarray(y_prob))
        if y_prob is not None else np.nan
    )
    return metrics


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    return metrics


def find_best_threshold_under_recall_constraint(
    y_true,
    y_prob,
    base_threshold=0.5,
    recall_tolerance=0.02,
    search_start=0.30,
    search_end=0.70,
    search_step=0.01,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    baseline_metrics = compute_metrics(
        y_true, y_prob, threshold=base_threshold
    )
    baseline_recall = baseline_metrics["recall"]
    thresholds = np.arange(
        search_start, search_end + 1e-12, search_step
    )
    best_thr = float(base_threshold)
    best_metrics = baseline_metrics.copy()
    found_valid_candidate = False
    for thr in thresholds:
        candidate_metrics = compute_metrics(
            y_true, y_prob, threshold=float(thr)
        )
        if candidate_metrics["recall"] < (
            baseline_recall - recall_tolerance
        ):
            continue
        if not found_valid_candidate:
            found_valid_candidate = True
            best_thr = float(thr)
            best_metrics = candidate_metrics
            continue
        better = False
        if candidate_metrics["f1"] > best_metrics["f1"]:
            better = True
        elif candidate_metrics["f1"] == best_metrics["f1"]:
            if candidate_metrics["accuracy"] > best_metrics["accuracy"]:
                better = True
            elif candidate_metrics["accuracy"] == best_metrics["accuracy"]:
                if abs(float(thr) - base_threshold) < abs(
                    best_thr - base_threshold
                ):
                    better = True
        if better:
            best_thr = float(thr)
            best_metrics = candidate_metrics
    return float(best_thr), best_metrics, baseline_metrics


def print_fold_summary(fold, metrics):
    print(
        f"[Fold {fold}] "
        f"ACC: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc']:.4f} | "
        f"SEN: {metrics['sensitivity']:.4f} | "
        f"SPEC: {metrics['specificity']:.4f}"
    )


def print_cv_summary(title, fold_results):
    metric_names = [
        "accuracy",
        "f1",
        "auc",
        "sensitivity",
        "specificity",
    ]
    print(f"\n{title}")
    for metric_name in metric_names:
        values = [result[metric_name] for result in fold_results]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        print(f"{metric_name:12s}: {mean_val:.5f} ± {std_val:.5f}")
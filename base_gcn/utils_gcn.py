# utils_gcn.py

import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# 랜덤 시드 고정
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 평균값 계산
class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


# AUC 계산
def safe_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan


# 분류 성능 평가
def compute_classification_metrics(
    y_true,
    y_pred,
    y_prob: Optional[List[float]] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        metrics["sensitivity"] = np.nan
        metrics["specificity"] = np.nan
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["auc"] = safe_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = np.nan
    return metrics


# 임계값 기반 성능 계산
def compute_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    return metrics


# Recall 제약 기반 임계값 탐색
def find_best_threshold_under_recall_constraint(
    y_true,
    y_prob,
    base_threshold: float = 0.5,
    recall_tolerance: float = 0.02,
    search_start: float = 0.30,
    search_end: float = 0.70,
    search_step: float = 0.01,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    baseline_metrics = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=base_threshold,
    )
    baseline_recall = baseline_metrics["recall"]
    thresholds = np.arange(search_start, search_end + 1e-12, search_step)
    best_thr = float(base_threshold)
    best_metrics = baseline_metrics.copy()
    found_valid_candidate = False
    for thr in thresholds:
        candidate_metrics = compute_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(thr),
        )
        if candidate_metrics["recall"] < (baseline_recall - recall_tolerance):
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
                if abs(float(thr) - base_threshold) < abs(best_thr - base_threshold):
                    better = True
        if better:
            best_thr = float(thr)
            best_metrics = candidate_metrics
    return float(best_thr), best_metrics, baseline_metrics


# 체크포인트 저장
def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_score: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if best_score is not None:
        checkpoint["best_score"] = float(best_score)
    if extra is not None:
        checkpoint["extra"] = extra
    torch.save(checkpoint, save_path)

# 체크포인트 불러오기
def load_checkpoint(
    load_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(
        load_path,
        map_location=map_location,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# 조기 종료
class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        if mode not in {"max", "min"}:
            raise ValueError(f"지원하지 않는 모드 : {mode}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
    def _is_improved(self, current_score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return current_score > (self.best_score + self.min_delta)
        return current_score < (self.best_score - self.min_delta)

    def step(self, current_score: float) -> bool:
        if self._is_improved(current_score):
            self.best_score = float(current_score)
            self.counter = 0
            self.should_stop = False
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


# 평가 지표 출력 형식
def format_metric_value(value: float) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{value:.4f}"


# 에포크별 학습 결과
def print_epoch_log(
    fold: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: Optional[float] = None,
) -> None:
    log_str = (
        f"[Fold {fold}] Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {format_metric_value(train_metrics['accuracy'])} | "
        f"Val Acc: {format_metric_value(val_metrics['accuracy'])} | "
        f"Train F1: {format_metric_value(train_metrics['f1'])} | "
        f"Val F1: {format_metric_value(val_metrics['f1'])} | "
        f"Val AUC: {format_metric_value(val_metrics['auc'])}"
    )
    if lr is not None:
        log_str += f" | LR: {lr:.6e}"
    print(log_str)


# 폴드별 최종 결과
def print_fold_summary(
    fold: int,
    metrics: Dict[str, float],
    best_epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
) -> None:
    print("=" * 60)
    print(f"[Fold {fold}] Summary")
    if best_epoch is not None:
        print(f"Best Epoch  : {best_epoch}")
    if best_val_loss is not None:
        print(f"Best ValLoss: {best_val_loss:.4f}")
    print(f"Accuracy    : {format_metric_value(metrics['accuracy'])}")
    print(f"F1-score    : {format_metric_value(metrics['f1'])}")
    print(f"AUC         : {format_metric_value(metrics['auc'])}")
    print(f"Sensitivity : {format_metric_value(metrics['sensitivity'])}")
    print(f"Specificity : {format_metric_value(metrics['specificity'])}")
    if "threshold" in metrics:
        print(f"Threshold   : {format_metric_value(metrics['threshold'])}")
    if "loss" in metrics:
        print(f"Loss        : {format_metric_value(metrics['loss'])}")
    print("=" * 60)

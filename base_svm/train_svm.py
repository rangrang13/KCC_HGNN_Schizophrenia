# train_svm.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config_ml import (
    SEED,
    NUM_FOLDS,
    USE_THRESHOLD_TUNING,
    BASE_THRESHOLD,
    RECALL_TOLERANCE,
    THRESHOLD_SEARCH_START,
    THRESHOLD_SEARCH_END,
    THRESHOLD_SEARCH_STEP,
    SVM_C,
    SVM_KERNEL,
    SVM_GAMMA,
    SVM_CLASS_WEIGHT,
    SVM_USE_STANDARD_SCALER,
)

from dataset_ml import load_dataset
from utils_ml import (
    set_seed,
    compute_metrics,
    find_best_threshold_under_recall_constraint,
    print_fold_summary,
    print_cv_summary,
)


# SVM 모델 구성
def build_svm_pipeline():
    steps = []
    if SVM_USE_STANDARD_SCALER:
        steps.append(("scaler", StandardScaler()))
    steps.append((
        "svm",
        SVC(
            C=SVM_C,
            kernel=SVM_KERNEL,
            gamma=SVM_GAMMA,
            class_weight=SVM_CLASS_WEIGHT,
            probability=True,
            random_state=SEED,
        ),
    ))
    return Pipeline(steps)


# 폴드별 학습 및 평가
def run_fold(fold, X, y, train_idx, val_idx):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    # 모델 학습
    model = build_svm_pipeline()
    model.fit(X_train, y_train)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    # 임계값 탐색
    if USE_THRESHOLD_TUNING:
        y_prob_train = model.predict_proba(X_train)[:, 1]
        best_thr, _, _ = find_best_threshold_under_recall_constraint(
            y_true=y_train,
            y_prob=y_prob_train,
            base_threshold=BASE_THRESHOLD,
            recall_tolerance=RECALL_TOLERANCE,
            search_start=THRESHOLD_SEARCH_START,
            search_end=THRESHOLD_SEARCH_END,
            search_step=THRESHOLD_SEARCH_STEP,
        )
    else:
        best_thr = BASE_THRESHOLD
    # 최종 검증 성능
    final_val_metrics = compute_metrics(
        y_true=y_val,
        y_prob=y_prob_val,
        threshold=best_thr,
    )
    print_fold_summary(fold, final_val_metrics)
    return final_val_metrics


# 전체 학습 실행
def main():
    set_seed(SEED)
    X, y, _ = load_dataset()
    # Stratified K-fold 교차검증
    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(y)), y), start=1
    ):
        fold_metrics = run_fold(
            fold=fold,
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        fold_results.append(fold_metrics)
    # 교차검증 결과 요약
    print_cv_summary(
        "SVM Cross-Validation Final Summary",
        fold_results,
    )


if __name__ == "__main__":
    main()

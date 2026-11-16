# dataset_ml.py

from pathlib import Path

import numpy as np

from config_ml import (
    CONTROL_DIR,
    PATIENT_DIR,
    EXCLUDE_FILES,
    LABEL_MAP,
    CORR_ROW_USE_ABS,
    CORR_ROW_ZERO_DIAG,
    CORR_ROW_ROWWISE_ZSCORE,
)


# 상관행렬 대칭화
def symmetrize_corr(corr: np.ndarray) -> np.ndarray:
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)

# 행별 Z-score 정규화
def rowwise_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    row_mean = x.mean(axis=1, keepdims=True)
    row_std = x.std(axis=1, keepdims=True)
    row_std = np.where(row_std < eps, 1.0, row_std)
    return (x - row_mean) / row_std

# 상관관계 특징 전처리
def preprocess_corr_row(corr: np.ndarray) -> np.ndarray:
    x = corr.copy()
    if CORR_ROW_USE_ABS:
        x = np.abs(x)
    if CORR_ROW_ZERO_DIAG:
        np.fill_diagonal(x, 0.0)
    if CORR_ROW_ROWWISE_ZSCORE:
        x = rowwise_zscore(x)
    if CORR_ROW_ZERO_DIAG:
        np.fill_diagonal(x, 0.0)
    return x.astype(np.float32)

# SVM 입력 특징 생성
def make_feature_from_corr(corr: np.ndarray) -> np.ndarray:
    corr = symmetrize_corr(corr)
    x = preprocess_corr_row(corr)
    return x.reshape(-1).astype(np.float32)

# COBRE 데이터셋 불러오기
def load_dataset():
    X = []
    y = []
    subject_ids = []
    for group_name, group_dir in [
        ("Control", CONTROL_DIR),
        ("Patient", PATIENT_DIR),
    ]:
        for file_path in sorted(Path(group_dir).rglob("*.npz")):
            if file_path.name in EXCLUDE_FILES:
                continue
            with np.load(file_path, allow_pickle=False) as data:
                corr = data["corr"]
            X.append(make_feature_from_corr(corr))
            y.append(LABEL_MAP[group_name])
            subject_ids.append(file_path.stem)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(subject_ids),
    )

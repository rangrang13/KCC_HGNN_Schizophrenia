# config_ml.py

import os
from pathlib import Path

# 데이터 경로
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("COBRE_DATA_DIR", str(PROJECT_DIR / "data")))
CONTROL_DIR = DATA_DIR / "Control"
PATIENT_DIR = DATA_DIR / "Patient"

# 제외할 파일
EXCLUDE_FILES = ["40075.npz"]

# 라벨 매핑
LABEL_MAP = {
    "Control": 0,
    "Patient": 1,
}

# 데이터 크기
NUM_ROIS = 200
EXPECTED_CORR_SHAPE = (200, 200)

# 상관관계 특징 설정
CORR_ROW_USE_ABS = False
CORR_ROW_ZERO_DIAG = True
CORR_ROW_ROWWISE_ZSCORE = True

# 입력 특징
FEATURE_MODE = "corr_row_flat"
INPUT_DIM = NUM_ROIS * NUM_ROIS

# 랜덤 시드
SEED = 42

# 교차검증
NUM_FOLDS = 5

# 임계값 탐색
USE_THRESHOLD_TUNING = True
BASE_THRESHOLD = 0.50
RECALL_TOLERANCE = 0.02
THRESHOLD_SEARCH_START = 0.30
THRESHOLD_SEARCH_END = 0.70
THRESHOLD_SEARCH_STEP = 0.01

# SVM 모델
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_USE_STANDARD_SCALER = True

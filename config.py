# config.py

import os

# 데이터 경로
DATA_DIR = "/Users/eunhye/PyCharmMiscProject/data/COBRE_SCZ/reorder_preprocessed/total_preprocessed_result/cpac_based/NoGSR/cc200"
CONTROL_DIR = os.path.join(DATA_DIR, "Control")
PATIENT_DIR = os.path.join(DATA_DIR, "Patient")

# 제외 대상 및 라벨
EXCLUDE_FILES = ["40075.npz"]
LABEL_MAP = {
    "Control": 0,
    "Patient": 1,
}

# 데이터 크기
NUM_ROIS = 200
TIME_LENGTH = 150
EXPECTED_TIME_SHAPE = (150, 200)
EXPECTED_CORR_SHAPE = (200, 200)

# 노드 특징 설정
NODE_FEATURE_MODE = "corr_row"
CORR_ROW_USE_ABS = False
CORR_ROW_ZERO_DIAG = True
INPUT_DIM = NUM_ROIS

# 하이퍼엣지 구성
USE_ABS_CORR = True
K_NEIGHBORS = 5
INCLUDE_SELF_IN_HYPEREDGE = True
NUM_HYPEREDGES_PER_SUBJECT = 200

# HGNN 모델 설정
NUM_CLASSES = 2
HIDDEN_DIM = 36
DROPOUT = 0.37
NUM_LAYERS = 1

# 학습 설정
SEED = 42
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 5e-4
WEIGHT_DECAY = 4e-3
EPOCHS = 100

# 조기 종료
USE_EARLY_STOPPING = True
PATIENCE = 8

# 교차검증
NUM_FOLDS = 5

# 임계값 탐색
USE_THRESHOLD_TUNING = True
BASE_THRESHOLD = 0.50
RECALL_TOLERANCE = 0.02
THRESHOLD_SEARCH_START = 0.30
THRESHOLD_SEARCH_END = 0.70
THRESHOLD_SEARCH_STEP = 0.01

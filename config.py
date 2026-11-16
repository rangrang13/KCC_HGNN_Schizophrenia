# config.py

import os

DATA_DIR = "/Users/eunhye/PyCharmMiscProject/data/COBRE_SCZ/reorder_preprocessed/total_preprocessed_result/cpac_based/NoGSR/cc200"
CONTROL_DIR = os.path.join(DATA_DIR, "Control")
PATIENT_DIR = os.path.join(DATA_DIR, "Patient")

# 제외할 파일
EXCLUDE_FILES = ["40075.npz"]

# 라벨 매핑
LABEL_MAP = {
    "Control": 0,
    "Patient": 1,
}

NUM_ROIS = 200
TIME_LENGTH = 150

EXPECTED_TIME_SHAPE = (150, 200)
EXPECTED_CORR_SHAPE = (200, 200)

NODE_FEATURE_MODE = "corr_row"

# corr_row feature 설정
CORR_ROW_USE_ABS = False
CORR_ROW_ZERO_DIAG = True

INPUT_DIM = NUM_ROIS

# corr 기반 hyperedge 생성 설정
USE_ABS_CORR = True
K_NEIGHBORS = 5
INCLUDE_SELF_IN_HYPEREDGE = True
NUM_HYPEREDGES_PER_SUBJECT = 200

# 클래스 수 (Control / Patient)
NUM_CLASSES = 2

HIDDEN_DIM = 36
DROPOUT = 0.37

# HGNN layer 수
NUM_LAYERS = 1

SEED = 42

# DHG batching 제약 때문에 batch_size=1 유지
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

LR = 5e-4
WEIGHT_DECAY = 4e-3
EPOCHS = 100

USE_EARLY_STOPPING = True
PATIENCE = 8

NUM_FOLDS = 5

USE_THRESHOLD_TUNING = True
BASE_THRESHOLD = 0.50

# train fold에서 threshold 탐색 시 recall이 baseline 대비 너무 많이 깨지지 않도록 허용하는 감소 폭
RECALL_TOLERANCE = 0.02

THRESHOLD_SEARCH_START = 0.30
THRESHOLD_SEARCH_END = 0.70
THRESHOLD_SEARCH_STEP = 0.01

DEVICE = "cpu"
CHECKPOINT_DIR = "../checkpoints"
SAVE_BEST_MODEL = True
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
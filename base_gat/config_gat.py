# config_gat.py

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
TIME_LENGTH = 150

EXPECTED_TIME_SHAPE = (150, 200)
EXPECTED_CORR_SHAPE = (200, 200)

# 노드 특징 
NODE_FEATURE_MODE = "corr_row"

# 상관관계 특징 설정
CORR_ROW_USE_ABS = False
CORR_ROW_ZERO_DIAG = True

INPUT_DIM = NUM_ROIS

# 그래프 구성
USE_ABS_CORR_FOR_GRAPH = True
K_NEIGHBORS = 5
INCLUDE_SELF_LOOP = True
MAKE_UNDIRECTED = True

# GAT 모델
HIDDEN_DIM = 36
DROPOUT = 0.37
POOLING = "mean"
NUM_LAYERS = 1
NUM_CLASSES = 2

GAT_HEADS = 4
GAT_CONCAT = False
GAT_NEGATIVE_SLOPE = 0.2
GAT_ADD_SELF_LOOPS = False
GAT_ATTENTION_DROPOUT = 0.37

# 랜덤 시드
SEED = 42

# 배치 설정
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# 학습 설정
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

# 실행 환경
DEVICE = "cpu"

# 모델 저장
CHECKPOINT_DIR = "../checkpoints_gat_pyg"
SAVE_BEST_MODEL = True
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# config.py

import os

# 데이터 경로
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

# 데이터 크기
NUM_ROIS = 200
TIME_LENGTH = 150

EXPECTED_TIME_SHAPE = (150, 200)
EXPECTED_CORR_SHAPE = (200, 200)

# 노드 특징 (corr_row / raw_time / summary)
NODE_FEATURE_MODE = "corr_row"

# 요약 특징 (평균 / 표준편차)
SUMMARY_USE_MEAN = True
SUMMARY_USE_STD = True

# 상관관계 특징 설정
CORR_ROW_USE_ABS = False
CORR_ROW_ZERO_DIAG = True

# 노드 특징별 입력 차원
def get_input_dim() -> int:
    if NODE_FEATURE_MODE == "corr_row":
        return NUM_ROIS
    if NODE_FEATURE_MODE == "raw_time":
        return TIME_LENGTH
    if NODE_FEATURE_MODE == "summary":
        dim = int(SUMMARY_USE_MEAN) + int(SUMMARY_USE_STD)
        return dim
    raise ValueError(f"지원하지 않는 노드 특징 모드 : {NODE_FEATURE_MODE}")


INPUT_DIM = get_input_dim()

# 하이퍼엣지 구성
USE_ABS_CORR = True
K_NEIGHBORS = 5
INCLUDE_SELF_IN_HYPEREDGE = True
NUM_HYPEREDGES_PER_SUBJECT = 200

# 클래스 수 (Control / Patient)
NUM_CLASSES = 2

# HGNN 모델 설정
HIDDEN_DIM = 36
DROPOUT = 0.37

# 풀링 방식 (mean / max)
POOLING = "mean"

# HGNN 레이어 수
NUM_LAYERS = 1

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
CHECKPOINT_DIR = "../checkpoints"
SAVE_BEST_MODEL = True
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
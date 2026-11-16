# config_gat.py

import os
import torch
from pathlib import Path

# 데이터 경로
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("COBRE_DATA_DIR", str(PROJECT_DIR / "data")))
CONTROL_DIR = DATA_DIR / "Control"
PATIENT_DIR = DATA_DIR / "Patient"

EXCLUDE_FILES = ["40075.npz"]

LABEL_MAP = {
    "Control": 0,
    "Patient": 1,
}

NUM_ROIS = 200
TIME_LENGTH = 150

EXPECTED_TIME_SHAPE = (150, 200)
EXPECTED_CORR_SHAPE = (200, 200)

NODE_FEATURE_MODE = "corr_row"

CORR_ROW_USE_ABS = False
CORR_ROW_ZERO_DIAG = True

INPUT_DIM = NUM_ROIS

# Graph Construction
USE_ABS_CORR_FOR_GRAPH = True
K_NEIGHBORS = 5
INCLUDE_SELF_LOOP = True
MAKE_UNDIRECTED = True

# GAT Model
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

# Training
SEED = 42

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

LR = 5e-4
WEIGHT_DECAY = 4e-3
EPOCHS = 100

USE_EARLY_STOPPING = True
PATIENCE = 8

NUM_FOLDS = 5

# Threshold Tuning
USE_THRESHOLD_TUNING = True
BASE_THRESHOLD = 0.50
RECALL_TOLERANCE = 0.02

THRESHOLD_SEARCH_START = 0.30
THRESHOLD_SEARCH_END = 0.70
THRESHOLD_SEARCH_STEP = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "../checkpoints_gat_pyg"

SAVE_BEST_MODEL = True

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

"""
Centralized configuration for the OCR pipeline.
"""
import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

# Checkpoints Directory
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'weights')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DET_MODEL_DIR = os.path.join(CHECKPOINT_DIR, 'dbnet')
REC_MODEL_DIR = os.path.join(CHECKPOINT_DIR, 'svtr')

# Default Best Weights
DEFAULT_DET_WEIGHTS = os.path.join(DET_MODEL_DIR, 'best.pth')
DEFAULT_REC_WEIGHTS = os.path.join(REC_MODEL_DIR, 'best.pth')

# ============================================================
# Inference Hyperparameters 
# ============================================================
DET_LIMIT_SIDE_LEN = 640      # Resize limit for detection
DET_DB_THRESH = 0.3           # DB binarization threshold
DET_DB_BOX_THRESH = 0.3       # DB box threshold (lower = more boxes)
# ============================================================
# Recognition (SVTR) Hyperparameters
# ============================================================
REC_IMAGE_SHAPE = (3, 32, 320) # (Channels, Height, Max Width)
REC_CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
REC_MEAN = [0.485, 0.456, 0.406]
REC_STD = [0.229, 0.224, 0.225]
REC_MIN_CROP_WIDTH = 8

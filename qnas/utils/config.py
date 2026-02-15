"""
Configuration management for QNAS (Quantum Neural Architecture Search).
Loads settings from environment variables with sensible defaults.
"""
import os
import sys
from typing import List
from pathlib import Path
from datetime import datetime


def _load_env_file(path: str = ".env"):
    """Load environment variables from .env file if it exists.
    
    Note: This uses setdefault() which only sets values if they don't already exist.
    To override an existing environment variable, unset it first or use a different mechanism.
    """
    env_path = Path(path)
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    # Use setdefault to allow environment variable overrides
                    # If you want .env to always win, use: os.environ[key] = value
                    os.environ.setdefault(key, value)


# Load .env file if it exists
_load_env_file()


def _env_get_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_get_list(key: str, default_list):
    val = os.environ.get(key, "")
    if not val:
        return default_list
    try:
        return [item.strip() for item in val.split(",") if item.strip()]
    except Exception:
        return default_list


def _env_get_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _parse_checkpoint_sizes(raw: str) -> List[int]:
    """Parse checkpoint sizes from comma-separated string."""
    sizes = []
    for part in raw.split(","):
        part = part.strip().lower()
        if part in ("full", "0", ""):
            sizes.append(0)
        else:
            try:
                sizes.append(int(part))
            except ValueError:
                pass
    return sizes


def _parse_checkpoint_epochs(raw: str) -> List[int]:
    """Parse target epochs from comma-separated string."""
    epochs = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                epochs.append(int(part))
            except ValueError:
                pass
    return epochs


# =========================
# NSGA-II Configuration
# =========================
POP_SIZE = _env_get_int("POP_SIZE", 12)
N_GEN = _env_get_int("N_GEN", 6)
SEED = _env_get_int("SEED", 35)
WORKERS_PER_GPU = _env_get_int("WORKERS_PER_GPU", 2)

# Training budgets
EVAL_EPOCHS = _env_get_int("EVAL_EPOCHS", 1)
MAX_TRAIN_BATCHES = _env_get_int("MAX_TRAIN_BATCHES", 20)
MAX_VAL_BATCHES = _env_get_int("MAX_VAL_BATCHES", 20)

# Final training
FINAL_TRAIN_EPOCHS = _env_get_int("FINAL_TRAIN_EPOCHS", 3)
FINAL_TRAIN_GPU = _env_get_int("FINAL_TRAIN_GPU", 0)
FINAL_TRAIN_GPUS = _env_get_list("FINAL_TRAIN_GPUS", [])  # Empty = use all available
PARETO_OBJECTIVES = _env_get_list("PARETO_OBJECTIVES", ["f1_1_minus_acc", "f2_circuit_cost"])

# =========================
# Dataset Configuration
# =========================
DATASET = _env_get_str("DATASET", "mnist").lower()
BATCH_SIZE = _env_get_int("BATCH_SIZE", 256)
TRAIN_SUBSET_SIZE = _env_get_int("TRAIN_SUBSET_SIZE", 16000)
VAL_SUBSET_SIZE = _env_get_int("VAL_SUBSET_SIZE", 3000)
DATA_ROOT = _env_get_str("DATA_ROOT", "./data")

FINAL_TRAIN_SUBSET_SIZE = _env_get_int("FINAL_TRAIN_SUBSET_SIZE", 0)
FINAL_VAL_SUBSET_SIZE = _env_get_int("FINAL_VAL_SUBSET_SIZE", 0)

# =========================
# Model Configuration
# =========================
ALLOWED_EMBEDDINGS = _env_get_list("ALLOWED_EMBEDDINGS", ["angle-x", "angle-y", "angle-z", "amplitude"])
PRE_CLASSICAL_LAYERS = _env_get_int("PRE_CLASSICAL_LAYERS", 1)
POST_CLASSICAL_LAYERS = _env_get_int("POST_CLASSICAL_LAYERS", 1)
CLASSICAL_HIDDEN_DIM = _env_get_int("CLASSICAL_HIDDEN_DIM", 64)
ANGLE_EMBEDDING_ACTIVATION = _env_get_str("ANGLE_EMBEDDING_ACTIVATION", "none").lower()

# Quantum circuit bounds
NQ_MIN = _env_get_int("NQ_MIN", 2)
NQ_MAX = _env_get_int("NQ_MAX", 12)
DEPTH_MIN = _env_get_int("DEPTH_MIN", 1)
DEPTH_MAX = _env_get_int("DEPTH_MAX", 6)
ERANGE_MIN = _env_get_int("ERANGE_MIN", 1)
ERANGE_MAX = _env_get_int("ERANGE_MAX", 4)
LR_MIN = _env_get_float("LR_MIN", 1e-3)
LR_MAX = _env_get_float("LR_MAX", 1e-1)

# CNOT modes
CNOT_MODES = ["all", "odd", "even", "none"]
CMODE_MIN = _env_get_int("CMODE_MIN", 0)
CMODE_MAX = _env_get_int("CMODE_MAX", 3)

# Quantum differentiation (0 = adjoint; >0 = shot-based with finite-diff, not supported for batched training)
SHOTS = _env_get_int("SHOTS", 0)
FINAL_SHOTS = _env_get_int("FINAL_SHOTS", 0)

# Wire cutting
CUT_TARGET_QUBITS = _env_get_int("CUT_TARGET_QUBITS", 0)

# DataLoader workers configuration
# Default: auto-detect (min(4, cpu_count-1)) on non-Windows, 0 on Windows
# Can be overridden via DATALOADER_NUM_WORKERS env var
import os as _os
_DEFAULT_DATALOADER_WORKERS = min(4, (_os.cpu_count() or 2) - 1) if _os.name != "nt" else 0
DATALOADER_NUM_WORKERS = _env_get_int("DATALOADER_NUM_WORKERS", _DEFAULT_DATALOADER_WORKERS)

# =========================
# Checkpoint Configuration
# =========================
CHECKPOINT_VALIDATION_ENABLED = _env_get_bool("CHECKPOINT_VALIDATION_ENABLED", False)
CHECKPOINT_NSGA_ENABLED = _env_get_bool("CHECKPOINT_NSGA_ENABLED", False)
CHECKPOINT_FINAL_ENABLED = _env_get_bool("CHECKPOINT_FINAL_ENABLED", False)
CHECKPOINT_CORRELATION_ENABLED = _env_get_bool("CHECKPOINT_CORRELATION_ENABLED", False)

CHECKPOINT_TRAIN_SIZES = _parse_checkpoint_sizes(_env_get_str("CHECKPOINT_TRAIN_SIZES", "2048,4096,8196,16392,32768,full"))
CHECKPOINT_TARGET_EPOCHS = _parse_checkpoint_epochs(_env_get_str("CHECKPOINT_TARGET_EPOCHS", "1,3,5,10"))

# =========================
# Prediction Configuration
# =========================
PREDICT_FINAL_ACC_ENABLED = _env_get_bool("PREDICT_FINAL_ACC_ENABLED", True)
PREDICT_FINAL_ACC_SLOPE = _env_get_float("PREDICT_FINAL_ACC_SLOPE", None)
PREDICT_FINAL_ACC_INTERCEPT = _env_get_float("PREDICT_FINAL_ACC_INTERCEPT", None)
PREDICT_FINAL_ACC_CHECKPOINT_FILE = _env_get_str("PREDICT_FINAL_ACC_CHECKPOINT_FILE", "")

# Aliases for prediction model (used by nsga2/problem.py)
PREDICTION_MODEL_ENABLED = PREDICT_FINAL_ACC_ENABLED
PREDICTION_SLOPE = PREDICT_FINAL_ACC_SLOPE if PREDICT_FINAL_ACC_SLOPE is not None else 0.95
PREDICTION_INTERCEPT = PREDICT_FINAL_ACC_INTERCEPT if PREDICT_FINAL_ACC_INTERCEPT is not None else 5.0
PREDICTION_MODEL_FILE = PREDICT_FINAL_ACC_CHECKPOINT_FILE

# =========================
# Logging Configuration
# =========================
LOG_DIR = _env_get_str("LOG_DIR", "./logs")
# Resolve to absolute path so logs go to a consistent location regardless of cwd
if not os.path.isabs(LOG_DIR):
    LOG_DIR = os.path.abspath(LOG_DIR)
RESUME_LOGS = _env_get_int("RESUME_LOGS", 0)

# Determine run type
IS_IMPORTED = os.environ.get("IMPORTED_AS_MODULE", "false").lower() == "true"
RUN_TYPE = os.environ.get("RUN_TYPE", "nsga" if not IS_IMPORTED else "correlation").lower()

# Setup logging directories
# Check if DATASET_LOG_DIR is already set (to prevent creating multiple run folders)
if "DATASET_LOG_DIR" in os.environ:
    # Use the existing directory (set by previous import or explicitly)
    DATASET_LOG_DIR = os.environ["DATASET_LOG_DIR"]
elif not IS_IMPORTED and RUN_TYPE == "nsga":
    # Create run folder when actually running NSGA-II (not when just importing)
    # Structure: logs/nsga-ii/{DATASET}/run_{TIMESTAMP}
    # This ensures each run has its own folder and prevents overwriting previous runs
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = f"run_{stamp}"
    nsga_base = os.path.join(LOG_DIR, "nsga-ii")
    # Use uppercase dataset name for folder (e.g., MNIST instead of mnist)
    dataset_folder_name = DATASET.upper()
    dataset_folder = os.path.join(nsga_base, dataset_folder_name)
    DATASET_LOG_DIR = os.path.join(dataset_folder, run_folder)
    # Store in environment to ensure consistency across imports
    os.environ["DATASET_LOG_DIR"] = DATASET_LOG_DIR
    # Create all parent directories if they don't exist
    os.makedirs(DATASET_LOG_DIR, exist_ok=True)
    
    # Copy .env to run folder as config_{stamp}.env for reproducibility (no .env in run dir)
    env_file = Path(".env")
    if env_file.exists():
        try:
            import shutil
            dest_env_named = Path(DATASET_LOG_DIR) / f"config_{stamp}.env"
            shutil.copy2(env_file, dest_env_named)
        except Exception as e:
            import sys
            print(f"[WARN] Could not copy .env to run folder: {e}", file=sys.stderr)
else:
    # When imported as a module or not running NSGA-II, don't create run folders
    # Use a temporary or existing directory - but don't default to LOG_DIR to avoid creating files there
    DATASET_LOG_DIR = os.path.join(LOG_DIR, "temp")
    # Don't create it - let the calling code specify where files should go

# Auto-enable checkpoint validation based on run type
if IS_IMPORTED and RUN_TYPE == "correlation":
    if not os.environ.get("CHECKPOINT_CORRELATION_ENABLED") and CHECKPOINT_CORRELATION_ENABLED:
        CHECKPOINT_VALIDATION_ENABLED = True
elif not IS_IMPORTED and RUN_TYPE == "nsga":
    # For NSGA-II, explicitly disable checkpoint validation unless CHECKPOINT_NSGA_ENABLED is True
    CHECKPOINT_VALIDATION_ENABLED = CHECKPOINT_NSGA_ENABLED

# =========================
# Worker Configuration
# =========================
# These are module-level variables that can be updated by worker processes
WORKER_GPU_ID = -1
WORKER_RANK = -1
STATUS_JSON_PATH = None


def _update_worker_info(gpu_id: int, rank: int):
    """Update worker identification info (called from nsga2/runner.py)."""
    global WORKER_GPU_ID, WORKER_RANK
    WORKER_GPU_ID = gpu_id
    WORKER_RANK = rank











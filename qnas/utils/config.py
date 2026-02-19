"""
Configuration management for QNAS (Quantum Neural Architecture Search).
Loads settings from environment variables with sensible defaults.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _load_env_file(path: str = ".env") -> None:
    """Load environment variables from .env file if it exists.

    Uses setdefault() so explicit environment variables override .env values.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            os.environ.setdefault(key, value)


# Load .env file if present.
_load_env_file()


def _env_get_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_get_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_get_float(key: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_get_list(key: str, default_list: List[str]) -> List[str]:
    raw = os.environ.get(key, "")
    if not raw:
        return default_list
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_get_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes", "on")


def _parse_checkpoint_sizes(raw: str) -> List[int]:
    """Parse checkpoint sizes from comma-separated string."""
    sizes: List[int] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if value in ("full", "0", ""):
            sizes.append(0)
            continue
        try:
            sizes.append(int(value))
        except ValueError:
            continue
    return sizes


def _parse_checkpoint_epochs(raw: str) -> List[int]:
    """Parse target epochs from comma-separated string."""
    epochs: List[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            epochs.append(int(value))
        except ValueError:
            continue
    return epochs


# =========================
# NSGA-II Configuration
# =========================
POP_SIZE = _env_get_int("POP_SIZE", 12)
N_GEN = _env_get_int("N_GEN", 6)
SEED = _env_get_int("SEED", 35)
WORKERS_PER_GPU = _env_get_int("WORKERS_PER_GPU", 2)

# Training budgets
EVAL_EPOCHS = _env_get_int("EVAL_EPOCHS", 2)
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
LR_MIN = _env_get_float("LR_MIN", 1e-3) or 1e-3
LR_MAX = _env_get_float("LR_MAX", 1e-1) or 1e-1

# CNOT modes
CNOT_MODES = ["all", "odd", "even", "none"]
CMODE_MIN = _env_get_int("CMODE_MIN", 0)
CMODE_MAX = _env_get_int("CMODE_MAX", 3)

# Quantum differentiation (0 = adjoint; >0 = shot-based)
SHOTS = _env_get_int("SHOTS", 0)
FINAL_SHOTS = _env_get_int("FINAL_SHOTS", 0)

# Wire cutting
CUT_TARGET_QUBITS = _env_get_int("CUT_TARGET_QUBITS", 0)

# DataLoader workers configuration
_DEFAULT_DATALOADER_WORKERS = min(4, (os.cpu_count() or 2) - 1) if os.name != "nt" else 0
DATALOADER_NUM_WORKERS = _env_get_int("DATALOADER_NUM_WORKERS", _DEFAULT_DATALOADER_WORKERS)
TRAIN_DROP_LAST = _env_get_bool("TRAIN_DROP_LAST", False)

# =========================
# Checkpoint Configuration
# =========================
CHECKPOINT_VALIDATION_ENABLED = _env_get_bool("CHECKPOINT_VALIDATION_ENABLED", False)
CHECKPOINT_NSGA_ENABLED = _env_get_bool("CHECKPOINT_NSGA_ENABLED", False)
CHECKPOINT_FINAL_ENABLED = _env_get_bool("CHECKPOINT_FINAL_ENABLED", False)
CHECKPOINT_CORRELATION_ENABLED = _env_get_bool("CHECKPOINT_CORRELATION_ENABLED", False)

CHECKPOINT_TRAIN_SIZES = _parse_checkpoint_sizes(
    _env_get_str("CHECKPOINT_TRAIN_SIZES", "2048,4096,8196,16392,32768,full")
)
CHECKPOINT_TARGET_EPOCHS = _parse_checkpoint_epochs(
    _env_get_str("CHECKPOINT_TARGET_EPOCHS", "1,3,5,10")
)

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
if not os.path.isabs(LOG_DIR):
    LOG_DIR = os.path.abspath(LOG_DIR)
RESUME_LOGS = _env_get_int("RESUME_LOGS", 0)

# Determine run type
IS_IMPORTED = os.environ.get("IMPORTED_AS_MODULE", "false").lower() == "true"
RUN_TYPE = os.environ.get("RUN_TYPE", "nsga" if not IS_IMPORTED else "correlation").lower()


def _build_nsga_run_dir(stamp: str) -> str:
    return os.path.join(LOG_DIR, "nsga-ii", DATASET.upper(), f"run_{stamp}")


def _copy_env_snapshot(run_dir: str, stamp: str) -> None:
    """Save a reproducibility snapshot of .env inside the run directory."""
    env_file = Path(".env")
    if not env_file.exists():
        return

    dest_env_named = Path(run_dir) / f"config_{stamp}.env"
    shutil.copy2(env_file, dest_env_named)


def _resolve_initial_dataset_log_dir() -> str:
    explicit = os.environ.get("DATASET_LOG_DIR", "").strip()
    if explicit:
        return os.path.abspath(explicit)
    # Keep imports side-effect free. Runtime entrypoints can initialize real run dirs.
    return os.path.join(LOG_DIR, "temp")


DATASET_LOG_DIR = _resolve_initial_dataset_log_dir()
os.environ.setdefault("DATASET_LOG_DIR", DATASET_LOG_DIR)


def set_dataset_log_dir(path: str, create: bool = False) -> str:
    """Set the active run directory used by logging utilities."""
    global DATASET_LOG_DIR

    resolved = os.path.abspath(path)
    if create:
        os.makedirs(resolved, exist_ok=True)

    DATASET_LOG_DIR = resolved
    os.environ["DATASET_LOG_DIR"] = resolved
    return DATASET_LOG_DIR


def initialize_nsga_run_dir(force_new: bool = False, copy_env_snapshot: bool = True) -> str:
    """Create or reuse an NSGA run directory explicitly at runtime.

    This function is intentionally not called during module import.
    """
    current = os.environ.get("DATASET_LOG_DIR", "").strip()
    if current and not force_new:
        current_abs = os.path.abspath(current)
        # Reuse explicit run directories, but treat default temp dir as non-final.
        if Path(current_abs).name.startswith("run_"):
            set_dataset_log_dir(current_abs, create=True)
            return DATASET_LOG_DIR

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _build_nsga_run_dir(stamp)
    set_dataset_log_dir(run_dir, create=True)

    if copy_env_snapshot:
        _copy_env_snapshot(run_dir, stamp)

    return DATASET_LOG_DIR


# Auto-enable checkpoint validation based on run type
if IS_IMPORTED and RUN_TYPE == "correlation":
    CHECKPOINT_VALIDATION_ENABLED = CHECKPOINT_CORRELATION_ENABLED
elif RUN_TYPE == "nsga":
    CHECKPOINT_VALIDATION_ENABLED = CHECKPOINT_NSGA_ENABLED

# =========================
# Worker Configuration
# =========================
WORKER_GPU_ID = -1
WORKER_RANK = -1
STATUS_JSON_PATH = None


def _update_worker_info(gpu_id: int, rank: int) -> None:
    """Update worker identification info (called from nsga2/runner.py)."""
    global WORKER_GPU_ID, WORKER_RANK
    WORKER_GPU_ID = gpu_id
    WORKER_RANK = rank

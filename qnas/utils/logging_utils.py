"""
Logging utilities for CSV files and progress logging.
"""
import os
import csv
import time
import json
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import DATASET_LOG_DIR, CHECKPOINT_VALIDATION_ENABLED, IS_IMPORTED, CNOT_MODES

# CSV file paths (can be overridden via environment)
default_nsga_csv = os.path.join(DATASET_LOG_DIR, "nsga_evals.csv")
default_epoch_csv = os.path.join(DATASET_LOG_DIR, "train_epoch_log.csv")
default_gen_csv = os.path.join(DATASET_LOG_DIR, "nsga_gen_summary.csv")
default_checkpoint_csv = os.path.join(DATASET_LOG_DIR, "checkpoint_validation.csv")

NSGA_EVAL_CSV = os.environ.get("NSGA_EVAL_CSV", default_nsga_csv)
EPOCH_LOG_CSV = os.environ.get("EPOCH_LOG_CSV", default_epoch_csv)
GEN_SUMMARY_CSV = os.environ.get("GEN_SUMMARY_CSV", default_gen_csv)
CHECKPOINT_LOG_CSV = os.environ.get("CHECKPOINT_LOG_CSV", default_checkpoint_csv)
default_progress_log = os.path.join(DATASET_LOG_DIR, "progress.log")
PROGRESS_LOG = os.environ.get("PROGRESS_LOG", default_progress_log)

STATUS_DIR = os.path.join(DATASET_LOG_DIR, "status")
if not IS_IMPORTED:
    os.makedirs(STATUS_DIR, exist_ok=True)

# CSV Headers
EVAL_HEADER = [
    "eval_id", "gen_est", "embed", "n_qubits", "depth", "ent_ranges", "cnot_modes", "learning_rate",
    "val_loss", "val_acc", "f1_1_minus_acc", "f2_circuit_cost", "f3_n_subcircuits",
    "q_backend", "seconds", "gpu_id", "worker_rank", "pid"
]

EPOCH_HEADER = [
    "eval_id", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "embed",
    "n_qubits", "depth", "ent_ranges", "cnot_modes", "lr", "q_backend", "shots",
    "gpu_id", "worker_rank", "pid", "phase", "batch", "batches_total", "elapsed_s"
]

GEN_HEADER = [
    "generation", "best_1_minus_acc", "best_cost", "best_n_subcircuits",
    "median_1_minus_acc", "elapsed_minutes"
]

CHECKPOINT_HEADER = [
    "eval_id", "epoch", "checkpoint_train_size", "val_loss", "val_acc", "embed",
    "n_qubits", "depth", "ent_ranges", "cnot_modes", "lr", "q_backend", "shots",
    "gpu_id", "worker_rank", "pid", "timestamp"
]

# Global locks and counters (set by NSGA-II runner)
CSV_LOCK = None
GLOBAL_COUNTER = None
CURRENT_GENERATION = None


def _get_worker_info():
    """Get current worker GPU ID and rank from config module."""
    from .config import WORKER_GPU_ID, WORKER_RANK
    return WORKER_GPU_ID, WORKER_RANK


# Module-level worker info (synced from config)
WORKER_GPU_ID = -1
WORKER_RANK = -1


def _csv_prepare(path: str, header: List[str]):
    """Create CSV file with header if it doesn't exist."""
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()
        except Exception:
            pass


def _csv_append(path: str, row: Dict[str, Any], header: List[str]):
    """Append a row to CSV file with thread-safe locking."""
    try:
        if CSV_LOCK is not None:
            try:
                with CSV_LOCK:
                    with open(path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
                return
            except Exception:
                pass
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
    except Exception:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
        except Exception:
            pass


def _append_progress(line: str):
    """Append a line to the progress log file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with open(PROGRESS_LOG, "a") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass


def _now_iso():
    """Get current time as ISO format string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _status_update(payload: Dict[str, Any]):
    """Update GPU status JSON file."""
    gpu_id, _ = _get_worker_info()
    if gpu_id < 0:
        return
    try:
        status_file = os.path.join(STATUS_DIR, f"worker_gpu{gpu_id}_status.json")
        payload["timestamp"] = _now_iso()
        with open(status_file, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def _csv_reset_all():
    """Reset all CSV files and progress log at the start of a run."""
    files = [
        (NSGA_EVAL_CSV, EVAL_HEADER),
        (EPOCH_LOG_CSV, EPOCH_HEADER),
        (GEN_SUMMARY_CSV, GEN_HEADER),
    ]
    if CHECKPOINT_VALIDATION_ENABLED:
        files.append((CHECKPOINT_LOG_CSV, CHECKPOINT_HEADER))
    
    for path, header in files:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()
        except Exception:
            pass
    
    # Reset progress log
    try:
        with open(PROGRESS_LOG, "w") as f:
            f.write("")
    except Exception:
        pass
    
    # Clear stale status files
    try:
        if os.path.isdir(STATUS_DIR):
            for fn in os.listdir(STATUS_DIR):
                if fn.endswith("_status.json") or fn.endswith(".tmp"):
                    with contextlib.suppress(Exception):
                        os.remove(os.path.join(STATUS_DIR, fn))
    except Exception:
        pass


# Initialize CSV files
_csv_prepare(NSGA_EVAL_CSV, EVAL_HEADER)
_csv_prepare(EPOCH_LOG_CSV, EPOCH_HEADER)
_csv_prepare(GEN_SUMMARY_CSV, GEN_HEADER)
if CHECKPOINT_VALIDATION_ENABLED:
    _csv_prepare(CHECKPOINT_LOG_CSV, CHECKPOINT_HEADER)


def _modes_str(modes) -> str:
    """Convert CNOT modes list to hyphen-separated string."""
    from .config import CNOT_MODES
    return "-".join(CNOT_MODES[int(m)] for m in modes)


def log_epoch(eval_id, epoch, tr_loss, tr_acc, va_loss, va_acc, cfg, backend,
              phase="epoch_end", batch_idx=None, batches_total=None, t0=None):
    """Log training epoch data to CSV.
    
    Args:
        eval_id: Evaluation ID
        epoch: Current epoch number
        tr_loss: Training loss
        tr_acc: Training accuracy
        va_loss: Validation loss (None if not available)
        va_acc: Validation accuracy (None if not available)
        cfg: QConfig object with model configuration
        backend: Quantum backend name
        phase: Training phase ('epoch_end', 'train_batch', etc.)
        batch_idx: Current batch index (optional)
        batches_total: Total number of batches (optional)
        t0: Start time for elapsed calculation (optional)
    """
    import time as _time
    import os as _os
    
    gpu_id, worker_rank = _get_worker_info()
    elapsed = (_time.time() - t0) if (t0 is not None) else None
    _csv_append(EPOCH_LOG_CSV, {
        "eval_id": eval_id, "epoch": epoch,
        "train_loss": f"{tr_loss:.6f}" if tr_loss is not None else "",
        "train_acc": f"{tr_acc:.4f}" if tr_acc is not None else "",
        "val_loss": f"{va_loss:.6f}" if va_loss is not None else "",
        "val_acc": f"{va_acc:.4f}" if va_acc is not None else "",
        "embed": cfg.embed_kind, "n_qubits": cfg.n_qubits, "depth": cfg.depth,
        "ent_ranges": "-".join(map(str, cfg.ent_ranges)) if cfg.ent_ranges else "",
        "cnot_modes": _modes_str(cfg.cnot_modes) if cfg.cnot_modes else "",
        "lr": f"{cfg.learning_rate:.6e}" if cfg.learning_rate else "",
        "q_backend": backend, "shots": (cfg.shots or 0),
        "gpu_id": gpu_id, "worker_rank": worker_rank, "pid": _os.getpid(),
        "phase": phase, "batch": batch_idx if batch_idx is not None else "",
        "batches_total": batches_total if batches_total is not None else "",
        "elapsed_s": f"{elapsed:.3f}" if elapsed is not None else ""
    }, EPOCH_HEADER)


def log_checkpoint(eval_id: str, epoch: int, checkpoint_size: int, va_loss, va_acc,
                   cfg, backend: str):
    """Log checkpoint validation results to CSV.
    
    Args:
        eval_id: Evaluation ID
        epoch: Epoch number
        checkpoint_size: Training data size at this checkpoint
        va_loss: Validation loss (can be None)
        va_acc: Validation accuracy (can be None)
        cfg: QConfig object with model configuration
        backend: Quantum backend name
    """
    import time as _time
    import os as _os
    
    if not CHECKPOINT_VALIDATION_ENABLED:
        return
    gpu_id, worker_rank = _get_worker_info()
    _csv_append(CHECKPOINT_LOG_CSV, {
        "eval_id": eval_id,
        "epoch": epoch,
        "checkpoint_train_size": checkpoint_size,
        "val_loss": f"{va_loss:.6f}" if va_loss is not None else "",
        "val_acc": f"{va_acc:.4f}" if va_acc is not None else "",
        "embed": cfg.embed_kind,
        "n_qubits": cfg.n_qubits,
        "depth": cfg.depth,
        "ent_ranges": "-".join(map(str, cfg.ent_ranges)) if cfg.ent_ranges else "",
        "cnot_modes": _modes_str(cfg.cnot_modes) if cfg.cnot_modes else "",
        "lr": f"{cfg.learning_rate:.6e}" if cfg.learning_rate else "",
        "q_backend": backend,
        "shots": (cfg.shots or 0),
        "gpu_id": gpu_id,
        "worker_rank": worker_rank,
        "pid": _os.getpid(),
        "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime())
    }, CHECKPOINT_HEADER)











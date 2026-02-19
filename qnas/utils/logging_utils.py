"""
Logging utilities for CSV files and progress logging.
"""

import contextlib
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as cfg

try:
    import fcntl  # Unix only
    HAS_FCNTL = True
except ImportError:
    fcntl = None
    HAS_FCNTL = False


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


# Paths are refreshed dynamically so imports stay side-effect free.
NSGA_EVAL_CSV = ""
EPOCH_LOG_CSV = ""
GEN_SUMMARY_CSV = ""
CHECKPOINT_LOG_CSV = ""
PROGRESS_LOG = ""
STATUS_DIR = ""

# Global locks/counters (set by NSGA-II runner)
CSV_LOCK = None
GLOBAL_COUNTER = None
CURRENT_GENERATION = None

# Fallback lock for platforms without fcntl
_THREAD_LOCK = threading.RLock()


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def _resolve_dataset_log_dir() -> str:
    env_dir = os.environ.get("DATASET_LOG_DIR", "").strip()
    if env_dir:
        resolved = os.path.abspath(env_dir)
        if resolved != cfg.DATASET_LOG_DIR:
            cfg.set_dataset_log_dir(resolved, create=False)
        return resolved

    return cfg.DATASET_LOG_DIR


def refresh_logging_paths(dataset_log_dir: Optional[str] = None) -> None:
    """Refresh runtime logging paths from env/config (no filesystem writes)."""
    global NSGA_EVAL_CSV, EPOCH_LOG_CSV, GEN_SUMMARY_CSV, CHECKPOINT_LOG_CSV, PROGRESS_LOG, STATUS_DIR

    if dataset_log_dir:
        cfg.set_dataset_log_dir(dataset_log_dir, create=False)

    base_dir = _resolve_dataset_log_dir()

    default_nsga_csv = os.path.join(base_dir, "nsga_evals.csv")
    default_epoch_csv = os.path.join(base_dir, "train_epoch_log.csv")
    default_gen_csv = os.path.join(base_dir, "nsga_gen_summary.csv")
    default_checkpoint_csv = os.path.join(base_dir, "checkpoint_validation.csv")
    default_progress_log = os.path.join(base_dir, "progress.log")

    NSGA_EVAL_CSV = os.environ.get("NSGA_EVAL_CSV", default_nsga_csv)
    EPOCH_LOG_CSV = os.environ.get("EPOCH_LOG_CSV", default_epoch_csv)
    GEN_SUMMARY_CSV = os.environ.get("GEN_SUMMARY_CSV", default_gen_csv)
    CHECKPOINT_LOG_CSV = os.environ.get("CHECKPOINT_LOG_CSV", default_checkpoint_csv)
    PROGRESS_LOG = os.environ.get("PROGRESS_LOG", default_progress_log)
    STATUS_DIR = os.path.join(base_dir, "status")


refresh_logging_paths()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


@contextlib.contextmanager
def _file_lock(path: str):
    """Cross-process lock using .lock files on Unix; thread lock fallback elsewhere."""
    lock_path = f"{path}.lock"
    _ensure_parent_dir(lock_path)

    if HAS_FCNTL:
        with open(lock_path, "a", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    else:
        with _THREAD_LOCK:
            yield


def _get_worker_info():
    """Get current worker GPU ID and rank from config module."""
    return cfg.WORKER_GPU_ID, cfg.WORKER_RANK


def _csv_prepare(path: str, header: List[str]) -> None:
    """Create CSV file with header if it doesn't exist."""
    refresh_logging_paths()
    with _file_lock(path):
        _ensure_parent_dir(path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()


def _csv_append(path: str, row: Dict[str, Any], header: List[str]) -> None:
    """Append a row to CSV file with inter-process-safe locking."""
    refresh_logging_paths()

    def _append_once() -> None:
        with _file_lock(path):
            _ensure_parent_dir(path)
            needs_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)

    try:
        if CSV_LOCK is not None:
            with CSV_LOCK:
                _append_once()
        else:
            _append_once()
    except Exception as exc:
        raise RuntimeError(f"Failed to append CSV row to {path}: {exc}") from exc


def _append_progress(line: str) -> None:
    """Append a line to the progress log file."""
    refresh_logging_paths()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with _file_lock(PROGRESS_LOG):
            _ensure_parent_dir(PROGRESS_LOG)
            with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {line}\n")
    except Exception as exc:
        _warn(f"Could not append to progress log ({PROGRESS_LOG}): {exc}")


def _now_iso() -> str:
    """Get current time as ISO format string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _status_update(payload: Dict[str, Any]) -> None:
    """Update GPU status JSON file."""
    refresh_logging_paths()
    gpu_id, _ = _get_worker_info()
    if gpu_id < 0:
        return

    status_file = os.path.join(STATUS_DIR, f"worker_gpu{gpu_id}_status.json")
    tmp_file = f"{status_file}.tmp"
    payload["timestamp"] = _now_iso()

    try:
        with _file_lock(status_file):
            os.makedirs(STATUS_DIR, exist_ok=True)
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, status_file)
    except Exception as exc:
        _warn(f"Could not update status file {status_file}: {exc}")
        with contextlib.suppress(Exception):
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


def _csv_reset_all() -> None:
    """Reset all CSV files and progress log at the start of a run."""
    refresh_logging_paths()

    files = [
        (NSGA_EVAL_CSV, EVAL_HEADER),
        (EPOCH_LOG_CSV, EPOCH_HEADER),
        (GEN_SUMMARY_CSV, GEN_HEADER),
    ]
    if cfg.CHECKPOINT_VALIDATION_ENABLED:
        files.append((CHECKPOINT_LOG_CSV, CHECKPOINT_HEADER))

    for path, header in files:
        with _file_lock(path):
            _ensure_parent_dir(path)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()

    with _file_lock(PROGRESS_LOG):
        _ensure_parent_dir(PROGRESS_LOG)
        with open(PROGRESS_LOG, "w", encoding="utf-8") as f:
            f.write("")

    os.makedirs(STATUS_DIR, exist_ok=True)
    for fn in os.listdir(STATUS_DIR):
        if fn.endswith("_status.json") or fn.endswith(".tmp"):
            with contextlib.suppress(Exception):
                os.remove(os.path.join(STATUS_DIR, fn))


def _modes_str(modes) -> str:
    """Convert CNOT modes list to hyphen-separated string."""
    return "-".join(cfg.CNOT_MODES[int(m)] for m in modes)


def log_epoch(eval_id, epoch, tr_loss, tr_acc, va_loss, va_acc, cfg_obj, backend,
              phase="epoch_end", batch_idx=None, batches_total=None, t0=None) -> None:
    """Log training epoch data to CSV."""
    import os as _os
    import time as _time

    refresh_logging_paths()

    gpu_id, worker_rank = _get_worker_info()
    elapsed = (_time.time() - t0) if (t0 is not None) else None

    _csv_append(EPOCH_LOG_CSV, {
        "eval_id": eval_id,
        "epoch": epoch,
        "train_loss": f"{tr_loss:.6f}" if tr_loss is not None else "",
        "train_acc": f"{tr_acc:.4f}" if tr_acc is not None else "",
        "val_loss": f"{va_loss:.6f}" if va_loss is not None else "",
        "val_acc": f"{va_acc:.4f}" if va_acc is not None else "",
        "embed": cfg_obj.embed_kind,
        "n_qubits": cfg_obj.n_qubits,
        "depth": cfg_obj.depth,
        "ent_ranges": "-".join(map(str, cfg_obj.ent_ranges)) if cfg_obj.ent_ranges else "",
        "cnot_modes": _modes_str(cfg_obj.cnot_modes) if cfg_obj.cnot_modes else "",
        "lr": f"{cfg_obj.learning_rate:.6e}" if cfg_obj.learning_rate else "",
        "q_backend": backend,
        "shots": (cfg_obj.shots or 0),
        "gpu_id": gpu_id,
        "worker_rank": worker_rank,
        "pid": _os.getpid(),
        "phase": phase,
        "batch": batch_idx if batch_idx is not None else "",
        "batches_total": batches_total if batches_total is not None else "",
        "elapsed_s": f"{elapsed:.3f}" if elapsed is not None else "",
    }, EPOCH_HEADER)


def log_checkpoint(eval_id: str, epoch: int, checkpoint_size: int, va_loss, va_acc,
                   cfg_obj, backend: str) -> None:
    """Log checkpoint validation results to CSV."""
    import os as _os

    refresh_logging_paths()

    if not cfg.CHECKPOINT_VALIDATION_ENABLED:
        return

    gpu_id, worker_rank = _get_worker_info()

    _csv_append(CHECKPOINT_LOG_CSV, {
        "eval_id": eval_id,
        "epoch": epoch,
        "checkpoint_train_size": checkpoint_size,
        "val_loss": f"{va_loss:.6f}" if va_loss is not None else "",
        "val_acc": f"{va_acc:.4f}" if va_acc is not None else "",
        "embed": cfg_obj.embed_kind,
        "n_qubits": cfg_obj.n_qubits,
        "depth": cfg_obj.depth,
        "ent_ranges": "-".join(map(str, cfg_obj.ent_ranges)) if cfg_obj.ent_ranges else "",
        "cnot_modes": _modes_str(cfg_obj.cnot_modes) if cfg_obj.cnot_modes else "",
        "lr": f"{cfg_obj.learning_rate:.6e}" if cfg_obj.learning_rate else "",
        "q_backend": backend,
        "shots": (cfg_obj.shots or 0),
        "gpu_id": gpu_id,
        "worker_rank": worker_rank,
        "pid": _os.getpid(),
        "timestamp": _now_iso(),
    }, CHECKPOINT_HEADER)

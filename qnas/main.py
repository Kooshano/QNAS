#!/usr/bin/env python3
"""
QNAS Main Entry Point.

This module provides the executable entrypoint and lazy backward-compatible
exports for legacy imports from ``qnas.main``.
"""

from importlib import import_module
from typing import Dict, Tuple


def main() -> None:
    """Main entry point for NSGA-II optimization."""
    from .utils import config as cfg

    # Explicit runtime initialization to keep imports side-effect free.
    cfg.initialize_nsga_run_dir(force_new=False, copy_env_snapshot=True)

    from .nsga2.runner import final_train, run_nsga2

    best = run_nsga2()
    final_train(best)


# Backward-compatibility lazy exports
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Entry points
    "run_nsga2": ("qnas.nsga2.runner", "run_nsga2"),
    "final_train": ("qnas.nsga2.runner", "final_train"),

    # Models
    "HybridQNN": ("qnas.models", "HybridQNN"),
    "QConfig": ("qnas.models", "QConfig"),

    # Datasets
    "get_dataloaders": ("qnas.utils.datasets", "get_dataloaders"),
    "make_subset": ("qnas.utils.datasets", "make_subset"),
    "IN_FEATURES": ("qnas.utils.datasets", "IN_FEATURES"),
    "N_CLASSES": ("qnas.utils.datasets", "N_CLASSES"),

    # Training
    "train_for_budget": ("qnas.training", "train_for_budget"),
    "evaluate": ("qnas.training", "evaluate"),
    "run_checkpoint_validation": ("qnas.training", "run_checkpoint_validation"),

    # Model I/O
    "save_model_weights": ("qnas.utils.model_io", "save_model_weights"),
    "load_model_weights": ("qnas.utils.model_io", "load_model_weights"),

    # Logging helpers
    "log_epoch": ("qnas.utils.logging_utils", "log_epoch"),
    "log_checkpoint": ("qnas.utils.logging_utils", "log_checkpoint"),
    "_csv_prepare": ("qnas.utils.logging_utils", "_csv_prepare"),
    "_csv_append": ("qnas.utils.logging_utils", "_csv_append"),
    "_append_progress": ("qnas.utils.logging_utils", "_append_progress"),
    "_status_update": ("qnas.utils.logging_utils", "_status_update"),
    "_csv_reset_all": ("qnas.utils.logging_utils", "_csv_reset_all"),
    "refresh_logging_paths": ("qnas.utils.logging_utils", "refresh_logging_paths"),
    "NSGA_EVAL_CSV": ("qnas.utils.logging_utils", "NSGA_EVAL_CSV"),
    "EPOCH_LOG_CSV": ("qnas.utils.logging_utils", "EPOCH_LOG_CSV"),
    "GEN_SUMMARY_CSV": ("qnas.utils.logging_utils", "GEN_SUMMARY_CSV"),
    "CHECKPOINT_LOG_CSV": ("qnas.utils.logging_utils", "CHECKPOINT_LOG_CSV"),
    "EVAL_HEADER": ("qnas.utils.logging_utils", "EVAL_HEADER"),
    "EPOCH_HEADER": ("qnas.utils.logging_utils", "EPOCH_HEADER"),
    "GEN_HEADER": ("qnas.utils.logging_utils", "GEN_HEADER"),
    "CHECKPOINT_HEADER": ("qnas.utils.logging_utils", "CHECKPOINT_HEADER"),
    "STATUS_DIR": ("qnas.utils.logging_utils", "STATUS_DIR"),
    "PROGRESS_LOG": ("qnas.utils.logging_utils", "PROGRESS_LOG"),

    # Quantum circuits
    "entangling_pairs": ("qnas.quantum.circuits", "entangling_pairs"),
    "_filter_pairs_by_mode": ("qnas.quantum.circuits", "_filter_pairs_by_mode"),
    "strongly_entangling_block": ("qnas.quantum.circuits", "strongly_entangling_block"),
    "make_q_device_lightning": ("qnas.quantum.circuits", "make_q_device_lightning"),
    "build_qnode_and_layer": ("qnas.quantum.circuits", "build_qnode_and_layer"),
    "EMBED_ALL": ("qnas.quantum.circuits", "EMBED_ALL"),
    "EMBED_TO_ROT": ("qnas.quantum.circuits", "EMBED_TO_ROT"),

    # Quantum metrics
    "circuit_cost": ("qnas.quantum.metrics", "circuit_cost"),
    "f3_num_subcircuits": ("qnas.quantum.metrics", "f3_num_subcircuits"),
    "_modes_str": ("qnas.quantum.metrics", "_modes_str"),
    "_qlayer_to_wirecut_string": ("qnas.quantum.metrics", "_qlayer_to_wirecut_string"),

    # NSGA-II helpers
    "QNNHyperProblem": ("qnas.nsga2.problem", "QNNHyperProblem"),
    "ProgressCallback": ("qnas.nsga2.callbacks", "ProgressCallback"),
    "predict_final_accuracy": ("qnas.nsga2.problem", "predict_final_accuracy"),

    # Configuration
    "POP_SIZE": ("qnas.utils.config", "POP_SIZE"),
    "N_GEN": ("qnas.utils.config", "N_GEN"),
    "SEED": ("qnas.utils.config", "SEED"),
    "WORKERS_PER_GPU": ("qnas.utils.config", "WORKERS_PER_GPU"),
    "EVAL_EPOCHS": ("qnas.utils.config", "EVAL_EPOCHS"),
    "MAX_TRAIN_BATCHES": ("qnas.utils.config", "MAX_TRAIN_BATCHES"),
    "MAX_VAL_BATCHES": ("qnas.utils.config", "MAX_VAL_BATCHES"),
    "FINAL_TRAIN_EPOCHS": ("qnas.utils.config", "FINAL_TRAIN_EPOCHS"),
    "FINAL_TRAIN_GPU": ("qnas.utils.config", "FINAL_TRAIN_GPU"),
    "FINAL_WORKERS_PER_GPU": ("qnas.utils.config", "FINAL_WORKERS_PER_GPU"),
    "DATASET": ("qnas.utils.config", "DATASET"),
    "BATCH_SIZE": ("qnas.utils.config", "BATCH_SIZE"),
    "TRAIN_SUBSET_SIZE": ("qnas.utils.config", "TRAIN_SUBSET_SIZE"),
    "VAL_SUBSET_SIZE": ("qnas.utils.config", "VAL_SUBSET_SIZE"),
    "DATA_ROOT": ("qnas.utils.config", "DATA_ROOT"),
    "FINAL_TRAIN_SUBSET_SIZE": ("qnas.utils.config", "FINAL_TRAIN_SUBSET_SIZE"),
    "FINAL_VAL_SUBSET_SIZE": ("qnas.utils.config", "FINAL_VAL_SUBSET_SIZE"),
    "ALLOWED_EMBEDDINGS": ("qnas.utils.config", "ALLOWED_EMBEDDINGS"),
    "PRE_CLASSICAL_LAYERS": ("qnas.utils.config", "PRE_CLASSICAL_LAYERS"),
    "POST_CLASSICAL_LAYERS": ("qnas.utils.config", "POST_CLASSICAL_LAYERS"),
    "CLASSICAL_HIDDEN_DIM": ("qnas.utils.config", "CLASSICAL_HIDDEN_DIM"),
    "ANGLE_EMBEDDING_ACTIVATION": ("qnas.utils.config", "ANGLE_EMBEDDING_ACTIVATION"),
    "NQ_MIN": ("qnas.utils.config", "NQ_MIN"),
    "NQ_MAX": ("qnas.utils.config", "NQ_MAX"),
    "DEPTH_MIN": ("qnas.utils.config", "DEPTH_MIN"),
    "DEPTH_MAX": ("qnas.utils.config", "DEPTH_MAX"),
    "ERANGE_MIN": ("qnas.utils.config", "ERANGE_MIN"),
    "ERANGE_MAX": ("qnas.utils.config", "ERANGE_MAX"),
    "LR_MIN": ("qnas.utils.config", "LR_MIN"),
    "LR_MAX": ("qnas.utils.config", "LR_MAX"),
    "CNOT_MODES": ("qnas.utils.config", "CNOT_MODES"),
    "CMODE_MIN": ("qnas.utils.config", "CMODE_MIN"),
    "CMODE_MAX": ("qnas.utils.config", "CMODE_MAX"),
    "SHOTS": ("qnas.utils.config", "SHOTS"),
    "FINAL_SHOTS": ("qnas.utils.config", "FINAL_SHOTS"),
    "CUT_TARGET_QUBITS": ("qnas.utils.config", "CUT_TARGET_QUBITS"),
    "CHECKPOINT_VALIDATION_ENABLED": ("qnas.utils.config", "CHECKPOINT_VALIDATION_ENABLED"),
    "CHECKPOINT_NSGA_ENABLED": ("qnas.utils.config", "CHECKPOINT_NSGA_ENABLED"),
    "CHECKPOINT_FINAL_ENABLED": ("qnas.utils.config", "CHECKPOINT_FINAL_ENABLED"),
    "CHECKPOINT_CORRELATION_ENABLED": ("qnas.utils.config", "CHECKPOINT_CORRELATION_ENABLED"),
    "CHECKPOINT_TRAIN_SIZES": ("qnas.utils.config", "CHECKPOINT_TRAIN_SIZES"),
    "CHECKPOINT_TARGET_EPOCHS": ("qnas.utils.config", "CHECKPOINT_TARGET_EPOCHS"),
    "LOG_DIR": ("qnas.utils.config", "LOG_DIR"),
    "RESUME_LOGS": ("qnas.utils.config", "RESUME_LOGS"),
    "DATASET_LOG_DIR": ("qnas.utils.config", "DATASET_LOG_DIR"),
    "IS_IMPORTED": ("qnas.utils.config", "IS_IMPORTED"),
    "RUN_TYPE": ("qnas.utils.config", "RUN_TYPE"),
    "WORKER_GPU_ID": ("qnas.utils.config", "WORKER_GPU_ID"),
    "WORKER_RANK": ("qnas.utils.config", "WORKER_RANK"),
    "STATUS_JSON_PATH": ("qnas.utils.config", "STATUS_JSON_PATH"),
    "initialize_nsga_run_dir": ("qnas.utils.config", "initialize_nsga_run_dir"),
    "set_dataset_log_dir": ("qnas.utils.config", "set_dataset_log_dir"),
    "PREDICTION_MODEL_ENABLED": ("qnas.utils.config", "PREDICTION_MODEL_ENABLED"),
    "PREDICTION_SLOPE": ("qnas.utils.config", "PREDICTION_SLOPE"),
    "PREDICTION_INTERCEPT": ("qnas.utils.config", "PREDICTION_INTERCEPT"),
    "PREDICTION_MODEL_FILE": ("qnas.utils.config", "PREDICTION_MODEL_FILE"),
}


__all__ = ["main", *sorted(_LAZY_EXPORTS.keys())]


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas.main' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if __name__ == "__main__":
    main()

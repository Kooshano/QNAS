#!/usr/bin/env python3
"""
QNAS Main Entry Point

This module serves as the main entry point for the NSGA-II optimization of
Hybrid Quantum Neural Networks. It also provides backward compatibility exports
for scripts that import from qnas.main directly.

Usage:
    python -m qnas.main              # Run NSGA-II optimization
    python qnas/main.py              # Alternative way to run
    
    # Or import for use in other scripts:
    from qnas.main import HybridQNN, QConfig, train_for_budget
"""

def main():
    """Main entry point for NSGA-II optimization."""
    # Import here to avoid issues when running as module
    from .nsga2.runner import run_nsga2, final_train
    
    best = run_nsga2()
    final_train(best)


# =========================
# Backward Compatibility Exports
# =========================
# These imports maintain backward compatibility for scripts that import from qnas.main
# Only loaded when this module is imported, not when run as __main__

# Entry points
from .nsga2.runner import run_nsga2, final_train

# Model classes and configuration
from .models import HybridQNN, QConfig

# Dataset utilities
from .utils.datasets import get_dataloaders, make_subset, IN_FEATURES, N_CLASSES

# Training functions
from .training import train_for_budget, evaluate, run_checkpoint_validation

# Model I/O utilities
from .utils.model_io import save_model_weights, load_model_weights

# Logging utilities
from .utils.logging_utils import (
    log_epoch, log_checkpoint,
    _csv_prepare, _csv_append, _append_progress, _status_update, _csv_reset_all,
    NSGA_EVAL_CSV, EPOCH_LOG_CSV, GEN_SUMMARY_CSV, CHECKPOINT_LOG_CSV,
    EVAL_HEADER, EPOCH_HEADER, GEN_HEADER, CHECKPOINT_HEADER,
    STATUS_DIR, PROGRESS_LOG
)

# Quantum circuit utilities
from .quantum.circuits import (
    entangling_pairs, _filter_pairs_by_mode, strongly_entangling_block,
    make_q_device_lightning, build_qnode_and_layer,
    EMBED_ALL, EMBED_TO_ROT
)

# Quantum metrics
from .quantum.metrics import (
    circuit_cost, f3_num_subcircuits, _modes_str, _qlayer_to_wirecut_string
)

# NSGA-II utilities
from .nsga2.problem import QNNHyperProblem, predict_final_accuracy
from .nsga2.callbacks import ProgressCallback

# Configuration (commonly accessed)
from .utils.config import (
    # NSGA-II config
    POP_SIZE, N_GEN, SEED, WORKERS_PER_GPU,
    EVAL_EPOCHS, MAX_TRAIN_BATCHES, MAX_VAL_BATCHES,
    FINAL_TRAIN_EPOCHS, FINAL_TRAIN_GPU,
    
    # Dataset config
    DATASET, BATCH_SIZE, TRAIN_SUBSET_SIZE, VAL_SUBSET_SIZE, DATA_ROOT,
    FINAL_TRAIN_SUBSET_SIZE, FINAL_VAL_SUBSET_SIZE,
    
    # Model config
    ALLOWED_EMBEDDINGS, PRE_CLASSICAL_LAYERS, POST_CLASSICAL_LAYERS,
    CLASSICAL_HIDDEN_DIM, ANGLE_EMBEDDING_ACTIVATION,
    NQ_MIN, NQ_MAX, DEPTH_MIN, DEPTH_MAX, ERANGE_MIN, ERANGE_MAX,
    LR_MIN, LR_MAX, CNOT_MODES, CMODE_MIN, CMODE_MAX,
    SHOTS, FINAL_SHOTS, CUT_TARGET_QUBITS,
    
    # Checkpoint config
    CHECKPOINT_VALIDATION_ENABLED, CHECKPOINT_NSGA_ENABLED,
    CHECKPOINT_FINAL_ENABLED, CHECKPOINT_CORRELATION_ENABLED,
    CHECKPOINT_TRAIN_SIZES, CHECKPOINT_TARGET_EPOCHS,
    
    # Logging config
    LOG_DIR, RESUME_LOGS, DATASET_LOG_DIR, IS_IMPORTED, RUN_TYPE,
    
    # Worker config
    WORKER_GPU_ID, WORKER_RANK, STATUS_JSON_PATH,
    
    # Prediction config
    PREDICTION_MODEL_ENABLED, PREDICTION_SLOPE, PREDICTION_INTERCEPT, PREDICTION_MODEL_FILE
)

# Module-level attributes for scripts that access them directly
# (maintained for backward compatibility with scripts like continue_training.py)
__all__ = [
    # Entry points
    'run_nsga2', 'final_train', 'main',
    
    # Models
    'HybridQNN', 'QConfig',
    
    # Datasets
    'get_dataloaders', 'make_subset', 'IN_FEATURES', 'N_CLASSES',
    
    # Training
    'train_for_budget', 'evaluate', 'run_checkpoint_validation',
    
    # Model I/O
    'save_model_weights', 'load_model_weights',
    
    # Logging
    'log_epoch', 'log_checkpoint', '_csv_append', '_append_progress', '_status_update',
    'NSGA_EVAL_CSV', 'EPOCH_LOG_CSV', 'GEN_SUMMARY_CSV', 'CHECKPOINT_LOG_CSV',
    'EVAL_HEADER', 'EPOCH_HEADER', 'GEN_HEADER', 'CHECKPOINT_HEADER',
    'STATUS_DIR', 'PROGRESS_LOG',
    
    # Quantum circuits
    'entangling_pairs', '_filter_pairs_by_mode', 'strongly_entangling_block',
    'make_q_device_lightning', 'build_qnode_and_layer', 'EMBED_ALL', 'EMBED_TO_ROT',
    
    # Quantum metrics
    'circuit_cost', 'f3_num_subcircuits', '_modes_str', '_qlayer_to_wirecut_string',
    
    # NSGA-II
    'QNNHyperProblem', 'ProgressCallback', 'predict_final_accuracy',
    
    # Config
    'POP_SIZE', 'N_GEN', 'SEED', 'WORKERS_PER_GPU',
    'EVAL_EPOCHS', 'MAX_TRAIN_BATCHES', 'MAX_VAL_BATCHES',
    'FINAL_TRAIN_EPOCHS', 'FINAL_TRAIN_GPU',
    'DATASET', 'BATCH_SIZE', 'TRAIN_SUBSET_SIZE', 'VAL_SUBSET_SIZE', 'DATA_ROOT',
    'FINAL_TRAIN_SUBSET_SIZE', 'FINAL_VAL_SUBSET_SIZE',
    'ALLOWED_EMBEDDINGS', 'PRE_CLASSICAL_LAYERS', 'POST_CLASSICAL_LAYERS',
    'CLASSICAL_HIDDEN_DIM', 'ANGLE_EMBEDDING_ACTIVATION',
    'NQ_MIN', 'NQ_MAX', 'DEPTH_MIN', 'DEPTH_MAX', 'ERANGE_MIN', 'ERANGE_MAX',
    'LR_MIN', 'LR_MAX', 'CNOT_MODES', 'CMODE_MIN', 'CMODE_MAX',
    'SHOTS', 'FINAL_SHOTS', 'CUT_TARGET_QUBITS',
    'CHECKPOINT_VALIDATION_ENABLED', 'CHECKPOINT_NSGA_ENABLED',
    'CHECKPOINT_FINAL_ENABLED', 'CHECKPOINT_CORRELATION_ENABLED',
    'CHECKPOINT_TRAIN_SIZES', 'CHECKPOINT_TARGET_EPOCHS',
    'LOG_DIR', 'RESUME_LOGS', 'DATASET_LOG_DIR', 'IS_IMPORTED', 'RUN_TYPE',
    'WORKER_GPU_ID', 'WORKER_RANK', 'STATUS_JSON_PATH',
]


if __name__ == "__main__":
    main()

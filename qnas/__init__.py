"""
QNAS — Quantum Neural Architecture Search

This package provides tools for optimizing Hybrid Quantum Neural Networks
using the NSGA-II multi-objective genetic algorithm.
"""
__version__ = "1.0.0"

# Backward compatibility exports - import from submodules directly
# This avoids the RuntimeWarning when running `python -m qnas.main`

# Entry points
from .nsga2.runner import run_nsga2, final_train

# Models
from .models import HybridQNN, QConfig

# Datasets
from .utils.datasets import get_dataloaders, make_subset, IN_FEATURES, N_CLASSES

# Training
from .training import train_for_budget, evaluate, run_checkpoint_validation

# Model I/O
from .utils.model_io import save_model_weights, load_model_weights

# Logging
from .utils.logging_utils import log_epoch, log_checkpoint

# Quantum circuits
from .quantum.circuits import entangling_pairs, _filter_pairs_by_mode, EMBED_ALL, EMBED_TO_ROT

# Quantum metrics
from .quantum.metrics import circuit_cost, _qlayer_to_wirecut_string

# NSGA-II
from .nsga2.problem import QNNHyperProblem
from .nsga2.callbacks import ProgressCallback

# Config
from .utils.config import DATASET, FINAL_SHOTS, DATASET_LOG_DIR, LOG_DIR

# Backward compatibility: Export modules for direct import
# This allows: from qnas import config, logging_utils, datasets
from .utils import config, logging_utils, datasets

__all__ = [
    # Entry points
    'run_nsga2', 'final_train',
    
    # Models
    'HybridQNN', 'QConfig',
    
    # Datasets
    'get_dataloaders', 'make_subset', 'IN_FEATURES', 'N_CLASSES',
    
    # Training
    'train_for_budget', 'evaluate', 'run_checkpoint_validation',
    
    # Model I/O
    'save_model_weights', 'load_model_weights',
    
    # Logging
    'log_epoch', 'log_checkpoint',
    
    # Quantum circuits
    'entangling_pairs', '_filter_pairs_by_mode', 'EMBED_ALL', 'EMBED_TO_ROT',
    
    # Quantum metrics
    'circuit_cost', '_qlayer_to_wirecut_string',
    
    # NSGA-II
    'QNNHyperProblem', 'ProgressCallback',
    
    # Config
    'DATASET', 'FINAL_SHOTS', 'DATASET_LOG_DIR', 'LOG_DIR',
]

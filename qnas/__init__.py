"""QNAS package root with lazy exports."""

from importlib import import_module
from typing import Dict, Tuple

__version__ = "1.1.0"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "run_nsga2": ("qnas.nsga2.runner", "run_nsga2"),
    "final_train": ("qnas.nsga2.runner", "final_train"),
    "HybridQNN": ("qnas.models", "HybridQNN"),
    "QConfig": ("qnas.models", "QConfig"),
    "QNNHyperProblem": ("qnas.nsga2.problem", "QNNHyperProblem"),
    "ProgressCallback": ("qnas.nsga2.callbacks", "ProgressCallback"),
    "predict_final_accuracy": ("qnas.nsga2.problem", "predict_final_accuracy"),
    "get_dataloaders": ("qnas.utils.datasets", "get_dataloaders"),
    "make_subset": ("qnas.utils.datasets", "make_subset"),
    "IN_FEATURES": ("qnas.utils.datasets", "IN_FEATURES"),
    "N_CLASSES": ("qnas.utils.datasets", "N_CLASSES"),
    "train_for_budget": ("qnas.training", "train_for_budget"),
    "evaluate": ("qnas.training", "evaluate"),
    "run_checkpoint_validation": ("qnas.training", "run_checkpoint_validation"),
    "save_model_weights": ("qnas.utils.model_io", "save_model_weights"),
    "load_model_weights": ("qnas.utils.model_io", "load_model_weights"),
    "log_epoch": ("qnas.utils.logging_utils", "log_epoch"),
    "log_checkpoint": ("qnas.utils.logging_utils", "log_checkpoint"),
    "entangling_pairs": ("qnas.quantum.circuits", "entangling_pairs"),
    "_filter_pairs_by_mode": ("qnas.quantum.circuits", "_filter_pairs_by_mode"),
    "EMBED_ALL": ("qnas.quantum.circuits", "EMBED_ALL"),
    "EMBED_TO_ROT": ("qnas.quantum.circuits", "EMBED_TO_ROT"),
    "circuit_cost": ("qnas.quantum.metrics", "circuit_cost"),
    "_qlayer_to_wirecut_string": ("qnas.quantum.metrics", "_qlayer_to_wirecut_string"),
    "DATASET": ("qnas.utils.config", "DATASET"),
    "FINAL_SHOTS": ("qnas.utils.config", "FINAL_SHOTS"),
    "DATASET_LOG_DIR": ("qnas.utils.config", "DATASET_LOG_DIR"),
    "LOG_DIR": ("qnas.utils.config", "LOG_DIR"),
}


__all__ = sorted([*_LAZY_EXPORTS.keys(), "config", "logging_utils", "datasets"])


def __getattr__(name: str):
    if name in {"config", "logging_utils", "datasets"}:
        module = import_module(f"qnas.utils.{name}")
        globals()[name] = module
        return module

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

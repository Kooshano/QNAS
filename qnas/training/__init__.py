"""Training utilities for Hybrid Quantum Neural Networks (lazy exports)."""

from importlib import import_module

_LAZY_EXPORTS = {
    "train_for_budget": ("qnas.training.trainer", "train_for_budget"),
    "evaluate": ("qnas.training.trainer", "evaluate"),
    "run_checkpoint_validation": ("qnas.training.checkpoint", "run_checkpoint_validation"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas.training' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

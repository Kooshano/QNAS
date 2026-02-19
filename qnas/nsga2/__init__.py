"""NSGA-II optimization namespace with lazy exports."""

from importlib import import_module

_LAZY_EXPORTS = {
    "QNNHyperProblem": ("qnas.nsga2.problem", "QNNHyperProblem"),
    "predict_final_accuracy": ("qnas.nsga2.problem", "predict_final_accuracy"),
    "run_nsga2": ("qnas.nsga2.runner", "run_nsga2"),
    "final_train": ("qnas.nsga2.runner", "final_train"),
    "ProgressCallback": ("qnas.nsga2.callbacks", "ProgressCallback"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas.nsga2' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

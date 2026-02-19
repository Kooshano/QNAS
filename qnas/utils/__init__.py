"""Utility modules for QNAS with lazy exports."""

from importlib import import_module

_LAZY_EXPORTS = {
    "save_model_weights": ("qnas.utils.model_io", "save_model_weights"),
    "load_model_weights": ("qnas.utils.model_io", "load_model_weights"),
}

__all__ = [
    "save_model_weights",
    "load_model_weights",
    "config",
    "logging_utils",
    "datasets",
    "model_io",
]


def __getattr__(name: str):
    if name in {"config", "logging_utils", "datasets", "model_io"}:
        module = import_module(f"qnas.utils.{name}")
        globals()[name] = module
        return module

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas.utils' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

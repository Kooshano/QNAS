"""Model definitions for Hybrid Quantum Neural Networks (lazy exports)."""

from importlib import import_module

_LAZY_EXPORTS = {
    "QConfig": ("qnas.models.config", "QConfig"),
    "HybridQNN": ("qnas.models.hybrid_qnn", "HybridQNN"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas.models' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

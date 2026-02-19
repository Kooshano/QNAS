"""Quantum circuit utilities (lazy exports)."""

from importlib import import_module

_LAZY_EXPORTS = {
    "entangling_pairs": ("qnas.quantum.circuits", "entangling_pairs"),
    "_filter_pairs_by_mode": ("qnas.quantum.circuits", "_filter_pairs_by_mode"),
    "strongly_entangling_block": ("qnas.quantum.circuits", "strongly_entangling_block"),
    "make_q_device_lightning": ("qnas.quantum.circuits", "make_q_device_lightning"),
    "build_qnode_and_layer": ("qnas.quantum.circuits", "build_qnode_and_layer"),
    "EMBED_TO_ROT": ("qnas.quantum.circuits", "EMBED_TO_ROT"),
    "EMBED_ALL": ("qnas.quantum.circuits", "EMBED_ALL"),
    "circuit_cost": ("qnas.quantum.metrics", "circuit_cost"),
    "f3_num_subcircuits": ("qnas.quantum.metrics", "f3_num_subcircuits"),
    "_modes_str": ("qnas.quantum.metrics", "_modes_str"),
    "_qlayer_to_wirecut_string": ("qnas.quantum.metrics", "_qlayer_to_wirecut_string"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qnas.quantum' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

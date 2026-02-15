"""Circuit metrics and cost calculation functions."""
import os
import sys
import contextlib
from typing import List

from .circuits import entangling_pairs, _filter_pairs_by_mode

# Import CNOT_MODES from config
try:
    from ..utils.config import CNOT_MODES
except ImportError:
    CNOT_MODES = ["all", "odd", "even", "none"]


# --- Helper to silence noisy C++/absl logs (stderr) during TF/cutter import & calls ---
@contextlib.contextmanager
def _suppress_stderr():
    """
    Hide low-level C++/absl logs that go straight to fd=2 (stderr). We use this
    only around the 'cutter' (TensorFlow) import and cut_placement() call.
    """
    try:
        fd = sys.stderr.fileno()
    except Exception:
        # Fallback: Python-level only
        old = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            yield
        finally:
            try:
                sys.stderr.close()
            except Exception:
                pass
            sys.stderr = old
        return

    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        try:
            os.dup2(saved, fd)
        finally:
            os.close(saved)
            os.close(devnull)


def _modes_str(modes: List[int]) -> str:
    """Convert CNOT modes list to hyphen-separated string.
    
    Args:
        modes: List of CNOT mode integers
        
    Returns:
        String representation (e.g., 'all-odd-none')
    """
    return "-".join(CNOT_MODES[int(m)] for m in modes)


def circuit_cost(n_qubits, depth, embed_kind, ent_ranges, cnot_modes):
    """Calculate the approximate circuit cost based on gate count.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of variational layers
        embed_kind: Type of embedding (not currently used in calculation)
        ent_ranges: List of entanglement ranges
        cnot_modes: List of CNOT modes
        
    Returns:
        Total gate count (single-qubit + two-qubit + embedding)
    """
    single = 3 * n_qubits * depth
    twoq = 0
    for l in range(depth):
        total_pairs = n_qubits
        mode = int(cnot_modes[l])
        if mode == 0:
            twoq += total_pairs
        elif mode == 1:
            twoq += (total_pairs + 1) // 2
        elif mode == 2:
            twoq += total_pairs // 2
        else:
            twoq += 0
    embedc = n_qubits
    return single + twoq + embedc


def _qlayer_to_wirecut_string(n_qubits: int, depth: int, ent_ranges: list, cnot_modes: list) -> str:
    """Build a minimal wire-cutting-compatible textual circuit: rx/ry/rz per wire, then CNOTs per layer.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of variational layers
        ent_ranges: List of entanglement ranges
        cnot_modes: List of CNOT modes
        
    Returns:
        QASM-like string representation of the circuit
    """
    lines = []
    ctr = 0
    for l in range(depth):
        for q in range(n_qubits):
            lines.append(f"rx({ctr}) q[{q}];")
            ctr += 1
            lines.append(f"ry({ctr}) q[{q}];")
            ctr += 1
            lines.append(f"rz({ctr}) q[{q}];")
            ctr += 1
        pairs = entangling_pairs(n_qubits, int(ent_ranges[l]))
        pairs = _filter_pairs_by_mode(pairs, int(cnot_modes[l]))
        for c, t in pairs:
            lines.append(f"cx q[{c}],q[{t}];")
    return "\n".join(lines)


# Lazy import of cut_placement to avoid TensorFlow import overhead until needed
_cut_placement_func = None
_cut_suppress_stderr = None


def f3_num_subcircuits(n_qubits: int, depth: int, ent_ranges: list, cnot_modes: list, cut_target_qubits: int) -> int:
    """Return number of sub-circuits for F3 using utils.cutter.cut_placement(...). 0 or None target -> 1.
    We *silence* TensorFlow/absl logs coming from cutter import & execution.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of variational layers
        ent_ranges: List of entanglement ranges
        cnot_modes: List of CNOT modes
        cut_target_qubits: Target number of qubits per subcircuit after cutting
    
    Returns:
        Number of subcircuits after wire cutting
        
    Raises:
        ImportError: If utils.cutter cannot be imported
        RuntimeError: If cut_placement fails to execute
    """
    if not cut_target_qubits or int(cut_target_qubits) <= 0:
        return 1
    global _cut_placement_func, _cut_suppress_stderr
    
    # Lazy import with stderr silenced to avoid TensorFlow cuDNN/cuBLAS factory spam
    if _cut_placement_func is None:
        try:
            with _suppress_stderr():
                from ..utils.cutter import cut_placement, _suppress_stderr as _cut_suppress_stderr_func
                _cut_placement_func = cut_placement
                _cut_suppress_stderr = _cut_suppress_stderr_func
        except ImportError as e:
            raise ImportError(
                f"Failed to import utils.cutter: {e}\n"
                f"This is required for F3 calculation. Please ensure utils/cutter.py is available and TensorFlow is installed."
            ) from e
    
    # Generate circuit string and apply cutting
    qtxt = _qlayer_to_wirecut_string(n_qubits, depth, ent_ranges, cnot_modes)
    try:
        with _cut_suppress_stderr():
            _, subwires_list = _cut_placement_func(qtxt, int(cut_target_qubits))
        return max(1, len(subwires_list))
    except Exception as e:
        raise RuntimeError(
            f"Failed to calculate F3 (number of subcircuits) using utils.cutter.cut_placement:\n"
            f"  Error: {e}\n"
            f"  Circuit: {n_qubits} qubits, depth {depth}, ent_ranges={ent_ranges}, cnot_modes={cnot_modes}\n"
            f"  Target qubits per subcircuit: {cut_target_qubits}"
        ) from e


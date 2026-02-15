"""Quantum circuit building functions for Hybrid Quantum Neural Networks."""
import torch
import pennylane as qml

# Embedding types
EMBED_ALL = ["angle-x", "angle-y", "angle-z", "amplitude"]
EMBED_TO_ROT = {"angle-x": "X", "angle-y": "Y", "angle-z": "Z"}


def entangling_pairs(n_qubits: int, r: int):
    """Generate entangling pairs for a given range.
    
    Args:
        n_qubits: Number of qubits
        r: Entanglement range
        
    Returns:
        List of qubit pairs for entanglement
    """
    r = int(max(1, min(r, n_qubits - 1)))
    return [(i, (i + r) % n_qubits) for i in range(n_qubits)]


def _filter_pairs_by_mode(pairs, mode_int: int):
    """Filter entangling pairs based on CNOT mode.
    
    Args:
        pairs: List of qubit pairs
        mode_int: CNOT mode (0=all, 1=odd, 2=even, 3=none)
        
    Returns:
        Filtered list of pairs
    """
    if mode_int == 0:
        return pairs           # all
    elif mode_int == 1:
        return [p for k, p in enumerate(pairs, start=1) if (k % 2) == 1]  # odd
    elif mode_int == 2:
        return [p for k, p in enumerate(pairs, start=1) if (k % 2) == 0]  # even
    else:
        return []              # none


def strongly_entangling_block(weights_l, n_qubits, ent_range, cnot_mode_int):
    """Apply a strongly entangling block to the quantum circuit.
    
    Args:
        weights_l: Weights for this layer (shape: [n_qubits, 3])
        n_qubits: Number of qubits
        ent_range: Entanglement range
        cnot_mode_int: CNOT mode
    """
    for q in range(n_qubits):
        rx, ry, rz = weights_l[q]
        qml.RX(rx, wires=q)
        qml.RY(ry, wires=q)
        qml.RZ(rz, wires=q)
    pairs = entangling_pairs(n_qubits, ent_range)
    pairs = _filter_pairs_by_mode(pairs, cnot_mode_int)
    for c, t in pairs:
        qml.CNOT(wires=[c, t])


def make_q_device_lightning(n_wires: int, shots: int):
    """Create a PennyLane quantum device with Lightning backend if available.
    
    Args:
        n_wires: Number of qubits
        shots: Number of shots (0 or None for adjoint differentiation)
        
    Returns:
        Tuple of (device, backend_name, diff_method)
    """
    use_adjoint = (shots is None) or (shots == 0)
    dev_shots = None if use_adjoint else int(shots)
    
    # With shots>0 we use finite-diff; PennyLane's finite-diff does not support broadcasted (batched)
    # tapes, so training (loss.backward()) will raise. Use shots=0 for training; shots>0 for eval only.
    if use_adjoint:
        diff_method = "adjoint"
    else:
        diff_method = "finite-diff"
    
    try:
        if torch.cuda.is_available():
            dev = qml.device("lightning.gpu", wires=n_wires, shots=dev_shots)
            return dev, "lightning.gpu", diff_method
    except Exception:
        pass
    try:
        dev = qml.device("lightning.qubit", wires=n_wires, shots=dev_shots)
        return dev, "lightning.qubit (CPU)", diff_method
    except Exception:
        dev = qml.device("default.qubit", wires=n_wires, shots=dev_shots)
        return dev, "default.qubit (CPU)", diff_method


def build_qnode_and_layer(n_qubits, depth, ent_ranges, cnot_modes, embed_kind, shots):
    """Build a PennyLane QNode and TorchLayer for the quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of variational layers
        ent_ranges: List of entanglement ranges for each layer
        cnot_modes: List of CNOT modes for each layer
        embed_kind: Type of embedding
        shots: Number of shots
        
    Returns:
        Tuple of (TorchLayer, backend_name)
    """
    dev, backend, diff_method = make_q_device_lightning(n_qubits, shots)
    weight_shapes = {"weights": (depth, n_qubits, 3)}
    embed_kind = embed_kind.lower()

    def circuit(inputs, weights):
        if embed_kind in EMBED_TO_ROT:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=EMBED_TO_ROT[embed_kind])
            # For angle-z, apply Hadamard gates to convert phase information to measurable amplitudes
            if embed_kind == "angle-z":
                # Hadamard gates convert |0⟩/|1⟩ (Z basis) to |+⟩/|-⟩ (X basis)
                # This makes phase information from RZ rotations observable
                for q in range(n_qubits):
                    qml.Hadamard(wires=q)
        else:
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=False)
        for l in range(depth):
            strongly_entangling_block(weights[l], n_qubits, ent_ranges[l], cnot_modes[l])
        
        # For angle-z, measure in X basis after Hadamard conversion
        # For amplitude, measure in Z basis (standard)
        if embed_kind == "angle-z":
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        else:
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method, cache=True)
    layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    return layer, backend


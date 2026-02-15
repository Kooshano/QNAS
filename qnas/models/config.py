"""Configuration dataclass for Hybrid Quantum Neural Network models."""
from dataclasses import dataclass


@dataclass
class QConfig:
    """Configuration for a Hybrid Quantum Neural Network.
    
    Attributes:
        embed_kind: Type of quantum embedding ('angle-x', 'angle-y', 'angle-z', 'amplitude')
        n_qubits: Number of qubits in the quantum circuit
        depth: Number of variational layers
        ent_ranges: List of entanglement ranges for each layer
        cnot_modes: List of CNOT modes for each layer (0=all, 1=odd, 2=even, 3=none)
        learning_rate: Learning rate for optimization
        shots: Number of shots for quantum simulation (0 for adjoint)
    """
    embed_kind: str
    n_qubits: int
    depth: int
    ent_ranges: list
    cnot_modes: list
    learning_rate: float
    shots: int


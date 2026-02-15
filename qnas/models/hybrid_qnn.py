"""Hybrid Quantum Neural Network model definition."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..quantum.circuits import build_qnode_and_layer, EMBED_TO_ROT
from ..utils.config import (
    PRE_CLASSICAL_LAYERS, POST_CLASSICAL_LAYERS, CLASSICAL_HIDDEN_DIM,
    ANGLE_EMBEDDING_ACTIVATION
)


class HybridQNN(nn.Module):
    """Hybrid Quantum-Classical Neural Network.
    
    Combines classical pre-processing layers, a variational quantum circuit,
    and classical post-processing layers for classification tasks.
    
    Args:
        n_qubits: Number of qubits in the quantum circuit
        depth: Number of variational layers
        ent_ranges: List of entanglement ranges for each layer
        cnot_modes: List of CNOT modes for each layer
        embed_kind: Type of embedding ('angle-x', 'angle-y', 'angle-z', 'amplitude')
        shots: Number of shots for quantum simulation (0 for adjoint)
        in_features: Number of input features
        n_classes: Number of output classes
    """
    
    def __init__(self, n_qubits, depth, ent_ranges, cnot_modes, embed_kind, shots, in_features: int, n_classes: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.ent_ranges = ent_ranges
        self.cnot_modes = cnot_modes
        self.embed_kind = embed_kind.lower()
        self.shots = shots
        self.in_features = in_features
        self.n_classes = n_classes
        self.vqc, self.q_backend = build_qnode_and_layer(n_qubits, depth, ent_ranges, cnot_modes, embed_kind, shots)
        
        # Pre-classical layers (before quantum circuit)
        self.pre_classical = nn.Sequential()
        current_dim = in_features
        
        if PRE_CLASSICAL_LAYERS > 0:
            for i in range(PRE_CLASSICAL_LAYERS):
                if i == 0:
                    # First layer: input -> hidden
                    self.pre_classical.add_module(f'pre_fc_{i}', nn.Linear(current_dim, CLASSICAL_HIDDEN_DIM))
                    self.pre_classical.add_module(f'pre_relu_{i}', nn.ReLU())
                    self.pre_classical.add_module(f'pre_dropout_{i}', nn.Dropout(0.1))
                    current_dim = CLASSICAL_HIDDEN_DIM
                else:
                    # Hidden layers
                    self.pre_classical.add_module(f'pre_fc_{i}', nn.Linear(current_dim, CLASSICAL_HIDDEN_DIM))
                    self.pre_classical.add_module(f'pre_relu_{i}', nn.ReLU())
                    self.pre_classical.add_module(f'pre_dropout_{i}', nn.Dropout(0.1))
                    current_dim = CLASSICAL_HIDDEN_DIM
        
        # Initialize quantum interface layers based on embedding type
        self.fc_angle = None
        self.fc_reduce = None  # For dimension reduction in amplitude embedding
        self.amp_temperature = None  # Temperature scaling for amplitude embedding
        
        if self.embed_kind in EMBED_TO_ROT:
            self.fc_angle = nn.Linear(current_dim, n_qubits)
        elif self.embed_kind == "amplitude":
            # Memory usage warning for high qubit counts
            if n_qubits > 4:
                print(f"WARNING: Amplitude embedding with {n_qubits} qubits requires {2**n_qubits} amplitudes")
                print(f"This may cause memory issues. Consider using angle embedding instead.")
            
            # Add specialized feature extraction for amplitude embedding
            target_dim = 2**n_qubits
            
            # For image data, use convolutional layers to extract spatial features
            # Only use conv layers if no pre-classical layers modify the dimension
            if in_features >= 784 and PRE_CLASSICAL_LAYERS == 0:  # MNIST or larger images without pre-processing
                self.amp_feature_extractor = nn.Sequential(
                    # Reshape will be done in forward pass
                    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # 28x28 -> 14x14
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),  # -> 4x4
                    nn.Flatten(),
                    nn.Linear(32 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, target_dim)
                )
                print(f"INFO: Added convolutional feature extractor for amplitude embedding")
            else:
                # For inputs with pre-classical layers or smaller datasets
                # Use enhanced feature extraction layers with residual connections
                if current_dim > target_dim:
                    # Multi-stage feature extraction with skip connections for better gradient flow
                    intermediate_dim = max(256, min(current_dim // 2, 1024))
                    self.amp_feature_extractor = nn.Sequential(
                        nn.Linear(current_dim, intermediate_dim),
                        nn.BatchNorm1d(intermediate_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(intermediate_dim, intermediate_dim),
                        nn.BatchNorm1d(intermediate_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(intermediate_dim, target_dim * 4),
                        nn.ReLU(),
                        nn.Dropout(0.05),
                        nn.Linear(target_dim * 4, target_dim * 2),
                        nn.ReLU(),
                        nn.Linear(target_dim * 2, target_dim)
                    )
                    print(f"INFO: Added enhanced feature extraction: {current_dim} → {intermediate_dim} → {target_dim}")
                else:
                    # Even for smaller inputs, use a small feature extractor
                    self.amp_feature_extractor = nn.Sequential(
                        nn.Linear(current_dim, target_dim * 2),
                        nn.ReLU(),
                        nn.Linear(target_dim * 2, target_dim)
                    )
                    print(f"INFO: Added feature extractor for amplitude: {current_dim} → {target_dim}")
            
            # Initialize learnable temperature parameter for amplitude embedding
            # Temperature scaling helps with gradient flow during training
            self.amp_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Post-classical layers (after quantum circuit)
        self.post_classical = nn.Sequential()
        current_dim = n_qubits  # Quantum circuit outputs n_qubits values
        
        if POST_CLASSICAL_LAYERS > 0:
            for i in range(POST_CLASSICAL_LAYERS):
                if i == POST_CLASSICAL_LAYERS - 1:
                    # Last layer: hidden -> output
                    self.post_classical.add_module(f'post_fc_{i}', nn.Linear(current_dim, n_classes))
                else:
                    # Hidden layers
                    self.post_classical.add_module(f'post_fc_{i}', nn.Linear(current_dim, CLASSICAL_HIDDEN_DIM))
                    self.post_classical.add_module(f'post_relu_{i}', nn.ReLU())
                    self.post_classical.add_module(f'post_dropout_{i}', nn.Dropout(0.1))
                    current_dim = CLASSICAL_HIDDEN_DIM
        else:
            # No post-classical layers, direct quantum -> output
            self.fc_out = nn.Linear(n_qubits, n_classes)

    def to_devices(self, device: torch.device):
        """Move model to specified device."""
        self.to(device)
        self.vqc.to(device)
        return self

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        
        # Apply pre-classical layers
        if PRE_CLASSICAL_LAYERS > 0:
            x = self.pre_classical(x)
        
        # Quantum processing
        if self.embed_kind in EMBED_TO_ROT:
            angles = self.fc_angle(x)
            
            # For angle-z (RZ rotations), we need full phase range [-π, π] 
            # RZ gates only modify phase, so restricting to [0, 2π) via ReLU limits expressivity
            # Always use tanh for angle-z to ensure proper [-π, π] phase range
            if self.embed_kind == "angle-z":
                # Use tanh to map to [-π, π] range for proper phase exploration
                # Scale by π (not 2π) to keep within reasonable range for Hadamard conversion
                # Hadamard gates work best when phase differences are in [-π, π]
                angles = math.pi * torch.tanh(angles)
            else:
                # For angle-x and angle-y, apply activation based on configuration
                if ANGLE_EMBEDDING_ACTIVATION == "relu":
                    angles = F.relu(angles)
                elif ANGLE_EMBEDDING_ACTIVATION == "tanh":
                    angles = math.pi * torch.tanh(angles)
                elif ANGLE_EMBEDDING_ACTIVATION == "sigmoid":
                    angles = math.pi * torch.sigmoid(angles)
                # else: "none" - no activation, allows full range including negatives
                
                # Wrap to [0, 2π) for angle-x and angle-y
                angles = torch.remainder(angles, 2*math.pi)
            
            z = self.vqc(angles)
        else:  # amplitude embedding
            # Use feature extractor if available
            if hasattr(self, 'amp_feature_extractor'):
                # Check if it's a conv-based extractor (only if no pre-classical layers and original input is image-sized)
                if self.in_features >= 784 and PRE_CLASSICAL_LAYERS == 0:
                    # Reshape to image format for conv layers
                    batch_size = x.size(0)
                    img_size = int(math.sqrt(self.in_features))  # Assumes square images
                    x_img = x.view(batch_size, 1, img_size, img_size)
                    feats = self.amp_feature_extractor(x_img)
                else:
                    # Use linear feature extractor (no reshaping needed)
                    feats = self.amp_feature_extractor(x)
            else:
                # Simple linear transformation
                feats = self.fc_amp(x)
            
            # Robust normalization for amplitude embedding with temperature scaling
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply temperature scaling to improve gradient flow
            # Higher temperature = softer distribution, better for learning
            if self.amp_temperature is not None:
                feats = feats / (self.amp_temperature + 1e-8)
            
            # Validate feature dimensions match expected amplitude count
            expected_dim = 2**self.n_qubits
            if feats.size(1) != expected_dim:
                raise ValueError(
                    f"Amplitude embedding dimension mismatch: expected {expected_dim} amplitudes "
                    f"(for {self.n_qubits} qubits), but got {feats.size(1)} features. "
                    f"This indicates a bug in the feature extractor."
                )
            
            # Normalize to unit vector (required for valid quantum state)
            # AmplitudeEmbedding expects real-valued amplitudes that sum to 1 in squared norm
            # We normalize using L2 norm (amplitudes can be negative, but |amplitude|^2 must sum to 1)
            norm = feats.norm(p=2, dim=1, keepdim=True)
            
            # Handle zero or invalid norms with better fallback
            bad = (norm < 1e-8).squeeze(1) | torch.isnan(norm.squeeze(1)) | torch.isinf(norm.squeeze(1))
            if bad.any():
                # For bad samples, use a small random initialization instead of uniform
                # This provides some diversity for the model to learn from
                amps = torch.randn_like(feats) * 0.1
                amps = amps / (amps.norm(p=2, dim=1, keepdim=True) + 1e-8)
            else:
                # Normalize to unit vector with small epsilon for numerical stability
                amps = feats / (norm + 1e-8)
            
            # Final validation: ensure amplitudes are valid (finite and correct shape)
            if torch.any(torch.isnan(amps)) or torch.any(torch.isinf(amps)):
                # Fallback to uniform superposition for invalid amplitudes
                amps = torch.ones_like(amps) / math.sqrt(expected_dim)
            
            z = self.vqc(amps)
        
        # Apply post-classical layers
        if POST_CLASSICAL_LAYERS > 0:
            out = self.post_classical(z)
        else:
            out = self.fc_out(z)
        
        return out


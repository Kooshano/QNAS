"""Model I/O utilities for saving and loading Hybrid Quantum Neural Network weights."""
import torch


def save_model_weights(model, weights_path: str, cfg, eval_id: str = None, 
                       epoch: int = None, val_acc: float = None, val_loss: float = None):
    """
    Save model weights with metadata (config, eval_id, epoch, etc.).
    Maintains backward compatibility - old code loading just state_dict will still work.
    
    Args:
        model: The HybridQNN model
        weights_path: Path to save weights
        cfg: QConfig with model configuration
        eval_id: Optional eval_id identifier
        epoch: Optional epoch number
        val_acc: Optional validation accuracy
        val_loss: Optional validation loss
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': {
            'embed_kind': cfg.embed_kind,
            'n_qubits': cfg.n_qubits,
            'depth': cfg.depth,
            'ent_ranges': cfg.ent_ranges,
            'cnot_modes': cfg.cnot_modes,
            'learning_rate': cfg.learning_rate,
            'shots': cfg.shots,
        },
        'metadata': {
            'eval_id': eval_id,
            'epoch': epoch,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'q_backend': model.q_backend if hasattr(model, 'q_backend') else None,
        }
    }
    torch.save(checkpoint, weights_path)


def load_model_weights(weights_path: str, map_location=None):
    """
    Load model weights, supporting both new format (with metadata) and old format (just state_dict).
    
    Args:
        weights_path: Path to the weights file
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
    
    Returns:
        tuple: (state_dict, config_dict, metadata_dict)
        - If old format: config_dict and metadata_dict will be None
        - If new format: all three will be populated
    """
    checkpoint = torch.load(weights_path, map_location=map_location)
    
    # Check if it's the new format (dict with 'state_dict' key) or old format (just state_dict)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # New format with metadata
        state_dict = checkpoint['state_dict']
        config = checkpoint.get('config', None)
        metadata = checkpoint.get('metadata', None)
        return state_dict, config, metadata
    else:
        # Old format - just state_dict
        return checkpoint, None, None


"""Training and evaluation functions for Hybrid Quantum Neural Networks."""
import os
import time
from typing import Optional, Tuple
from tqdm import tqdm

import torch
from torch import nn

from ..models.hybrid_qnn import HybridQNN
from ..models.config import QConfig
from ..utils.datasets import get_dataloaders, IN_FEATURES, N_CLASSES
from ..utils.logging_utils import _status_update, _append_progress, log_epoch, log_checkpoint, _get_worker_info
from ..quantum.metrics import _modes_str
from ..utils.config import (
    BATCH_SIZE,
    CHECKPOINT_VALIDATION_ENABLED, CHECKPOINT_TRAIN_SIZES,
    CHECKPOINT_FINAL_ENABLED, CHECKPOINT_CORRELATION_ENABLED,
    CHECKPOINT_NSGA_ENABLED, CHECKPOINT_TARGET_EPOCHS
)
from .checkpoint import run_checkpoint_validation


def _get_gpu_id():
    """Get current worker GPU ID."""
    gpu_id, _ = _get_worker_info()
    return gpu_id


def evaluate(model, loss_fn, loader, cfg: QConfig, max_batches=None, epoch=0, tag="Eval", eval_id=""):
    """Evaluate model on validation data.
    
    Args:
        model: The HybridQNN model
        loss_fn: Loss function
        loader: Validation data loader
        cfg: Model configuration
        max_batches: Maximum number of batches to evaluate (None for all)
        epoch: Current epoch number
        tag: Tag for logging
        eval_id: Evaluation ID for logging
    
    Returns:
        tuple: (loss, accuracy, elapsed_seconds, samples_seen)
            - loss: average validation loss
            - accuracy: validation accuracy in percentage
            - elapsed_seconds: pure evaluation time (GPU-synchronized, no process switching overhead)
            - samples_seen: number of samples processed
    
    Note: elapsed_seconds is measured with GPU synchronization to ensure accurate timing
    that can be compared across different configurations without process switching effects.
    """
    model.eval()
    device = next(model.parameters()).device
    run_loss, correct, seen, it = 0.0, 0, 0, 0
    total_batches = len(loader) if not max_batches else min(len(loader), max_batches)
    pbar = tqdm(loader, desc=f"[GPU{_get_gpu_id()}|{eval_id}] {tag} (epoch {epoch})", leave=False)
    
    # Synchronize GPU before timing to ensure all previous operations are complete
    # This ensures we measure pure evaluation time without interference from prior async ops
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t0_eval = time.time()
    
    for images, labels in pbar:
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                out = model(images)
                loss = loss_fn(out, labels)
            bs = images.size(0)
            run_loss += loss.item() * bs
            correct  += (out.argmax(1) == labels).sum().item()
            seen     += bs
            it += 1
            pbar.set_postfix(loss=f"{run_loss/max(1,seen):.4f}", acc=f"{100.0*correct/max(1,seen):.2f}%")
            # Keep live status updates for the watcher...
            _status_update({"stage": "val", "eval_id": eval_id, "epoch": epoch,
                            "batch": it, "batches_total": total_batches,
                            "val_loss_running": run_loss/max(1,seen), "val_acc_running": 100.0*correct/max(1,seen)})
            # ...but DO NOT write a CSV row per validation batch anymore (to avoid duplicate per-epoch lines).
            
            # Clean up intermediate tensors (but don't empty cache here to avoid timing overhead)
            del out, loss, images, labels
            
            if max_batches and it >= max_batches: 
                break
        except RuntimeError as e:
            error_msg = str(e)
            # Check for CUDA/GPU memory errors
            if "CUDA" in error_msg or "out of memory" in error_msg.lower() or "custatevec" in error_msg.lower():
                _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] GPU memory error during evaluation at epoch {epoch}, batch {it}: {error_msg}")
                # Clean up and try to recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # Skip this configuration - it's too large for available memory
                raise RuntimeError(f"GPU memory allocation failed during evaluation for config: embed={cfg.embed_kind}, nq={cfg.n_qubits}, depth={cfg.depth}") from e
            else:
                # Re-raise non-memory errors
                raise
    
    # Synchronize GPU after evaluation to ensure all async operations are complete
    # This gives us accurate end-to-end evaluation time for the circuit
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_seconds = time.time() - t0_eval
    
    # Clean up GPU memory after timing measurement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # CSV logging happens once per epoch in train_for_budget(...), phase="epoch_end".
    return run_loss / max(1, seen), 100.0 * correct / max(1, seen), elapsed_seconds, seen


def train_for_budget(cfg: QConfig, eval_id: str, epochs: int, max_train_batches: int, max_val_batches: int, 
                     train_size: Optional[int] = None, val_size: Optional[int] = None) -> Tuple[float, float, HybridQNN, float, int]:
    """Train a model for a given configuration and budget.
    
    Args:
        cfg: Model configuration
        eval_id: Evaluation ID for logging
        epochs: Number of training epochs
        max_train_batches: Maximum training batches per epoch (0 for all)
        max_val_batches: Maximum validation batches (0 for all)
        train_size: Training subset size (None for full dataset)
        val_size: Validation subset size (None for full dataset)
    
    Returns:
        tuple: (val_loss, val_acc, model, val_time, val_samples)
    """
    in_pool_worker = (os.environ.get("QNAS_POOL_WORKER", "0") == "1")
    train_loader, val_loader = get_dataloaders(in_pool_worker=in_pool_worker, train_size=train_size, val_size=val_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HybridQNN(cfg.n_qubits, cfg.depth, cfg.ent_ranges, cfg.cnot_modes, cfg.embed_kind, cfg.shots, IN_FEATURES, N_CLASSES).to_devices(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Determine if checkpoint validation should run based on eval_id prefix
    should_run_checkpoints = False
    if CHECKPOINT_VALIDATION_ENABLED and CHECKPOINT_TRAIN_SIZES:
        if eval_id.startswith("final-best") and CHECKPOINT_FINAL_ENABLED:
            should_run_checkpoints = True
        elif eval_id.startswith("final-") and CHECKPOINT_CORRELATION_ENABLED:
            should_run_checkpoints = True
        elif not eval_id.startswith("final-") and CHECKPOINT_NSGA_ENABLED:
            should_run_checkpoints = True
    
    # Run epoch 1 checkpoint validation if enabled (with multiple data sizes)
    if should_run_checkpoints:
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Checkpoint validation enabled at epoch 1")
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Config: embed={cfg.embed_kind}, nq={cfg.n_qubits}, depth={cfg.depth}, "
                         f"ranges={cfg.ent_ranges}, cnot={_modes_str(cfg.cnot_modes)}, lr={cfg.learning_rate:.2e}, backend={model.q_backend}")
        _status_update({"stage": "checkpoint_validation_epoch1", "eval_id": eval_id, "config": {
            "embed": cfg.embed_kind, "n_qubits": cfg.n_qubits, "depth": cfg.depth,
            "ent_ranges": cfg.ent_ranges, "cnot_modes": cfg.cnot_modes, "lr": cfg.learning_rate,
            "backend": model.q_backend, "shots": cfg.shots
        }})
        
        # Get the full training dataset size for reference
        full_train_dataset, _ = get_dataloaders(in_pool_worker=in_pool_worker, train_size=None, val_size=None)
        full_train_size = len(full_train_dataset.dataset) if hasattr(full_train_dataset, 'dataset') else 0
        
        # Run checkpoint validation at epoch 1 with all data sizes
        run_checkpoint_validation(model, loss_fn, cfg, eval_id, 
                                 CHECKPOINT_TRAIN_SIZES, in_pool_worker, full_train_size)
        
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Epoch 1 checkpoint validation complete. Continuing to train for remaining epochs...")
    
    # Normal training path (for all epochs, or epochs 2+ if checkpoints were run at epoch 1)
    # Determine starting epoch (skip epoch 1 if checkpoints already ran)
    start_epoch = 2 if should_run_checkpoints else 1
    
    if not should_run_checkpoints:
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Start training | embed={cfg.embed_kind}, nq={cfg.n_qubits}, depth={cfg.depth}, "
                         f"ranges={cfg.ent_ranges}, cnot={_modes_str(cfg.cnot_modes)}, lr={cfg.learning_rate:.2e}, backend={model.q_backend}")
        _status_update({"stage": "train_start", "eval_id": eval_id, "config": {
            "embed": cfg.embed_kind, "n_qubits": cfg.n_qubits, "depth": cfg.depth,
            "ent_ranges": cfg.ent_ranges, "cnot_modes": cfg.cnot_modes, "lr": cfg.learning_rate,
            "backend": model.q_backend, "shots": cfg.shots
        }})
    else:
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Continuing training from epoch {start_epoch} to {epochs}")

    vloss, vacc, val_time, val_samples = 0.0, 0.0, 0.0, 0
    # Best validation accuracy so far (and its loss) for checkpoint CSV: log best-up-to-this-epoch, not current
    best_val_acc, best_val_loss = -1.0, None

    for ep in range(start_epoch, epochs + 1):
        model.train()
        run_loss, correct, seen, it = 0.0, 0, 0, 0
        total_batches = len(train_loader) if not max_train_batches else min(len(train_loader), max_train_batches)
        pbar = tqdm(train_loader, desc=f"[GPU{_get_gpu_id()}|{eval_id}] Train {ep}/{epochs}", leave=False)
        t0_epoch = time.time()
        for images, labels in pbar:
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(images)
                loss = loss_fn(out, labels)
                loss.backward()
                opt.step()
                bs = images.size(0)
                run_loss += loss.item() * bs
                correct  += (out.argmax(1) == labels).sum().item()
                seen     += bs
                it += 1
                pbar.set_postfix(loss=f"{run_loss/max(1,seen):.4f}",
                                 acc=f"{100.0*correct/max(1,seen):.2f}%",
                                 lr=f"{opt.param_groups[0]['lr']:.1e}")
                _status_update({"stage": "train", "eval_id": eval_id, "epoch": ep,
                                "batch": it, "batches_total": total_batches,
                                "train_loss_running": run_loss/max(1,seen), "train_acc_running": 100.0*correct/max(1,seen)})
                log_epoch(eval_id, ep, run_loss/max(1,seen), 100.0*correct/max(1,seen), None, None,
                          cfg, model.q_backend, phase="train_batch", batch_idx=it,
                          batches_total=total_batches, t0=t0_epoch)
                
                # Clean up intermediate tensors
                del out, loss, images, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if max_train_batches and it >= max_train_batches: break
            except RuntimeError as e:
                error_msg = str(e)
                # Check for CUDA/GPU memory errors
                if "CUDA" in error_msg or "out of memory" in error_msg.lower() or "custatevec" in error_msg.lower():
                    _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] GPU memory error at epoch {ep}, batch {it}: {error_msg}")
                    # Clean up and try to recover
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # Skip this configuration - it's too large for available memory
                    raise RuntimeError(f"GPU memory allocation failed for config: embed={cfg.embed_kind}, nq={cfg.n_qubits}, depth={cfg.depth}") from e
                else:
                    # Re-raise non-memory errors
                    raise

        tr_loss = run_loss / max(1, seen)
        tr_acc  = 100.0 * correct / max(1, seen)
        vloss, vacc, val_time, val_samples = evaluate(model, loss_fn, val_loader, cfg, max_batches=max_val_batches, epoch=ep, tag=f"Val", eval_id=eval_id)
        # Track best validation accuracy so far (and its loss) for checkpoint CSV: write best-up-to-this-epoch
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_val_loss = vloss
        if best_val_loss is None:
            best_val_loss = vloss
        log_epoch(eval_id, ep, tr_loss, tr_acc, vloss, vacc, cfg, model.q_backend, phase="epoch_end", t0=t0_epoch)
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Ep {ep}/{epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | val_loss={vloss:.4f} acc={vacc:.2f}%")
        
        # Log to checkpoint validation CSV at target epochs (with full dataset only)
        # Log best val_acc so far (up to this epoch), not current epoch's acc (e.g. if epoch 5 < epoch 4, log epoch 4's acc)
        if should_run_checkpoints and ep in CHECKPOINT_TARGET_EPOCHS:
            # Get full training size for logging
            full_train_dataset, _ = get_dataloaders(in_pool_worker=in_pool_worker, train_size=None, val_size=None)
            full_train_size = len(full_train_dataset.dataset) if hasattr(full_train_dataset, 'dataset') else 0
            log_checkpoint(eval_id, epoch=ep, checkpoint_size=full_train_size, va_loss=best_val_loss,
                          va_acc=best_val_acc, cfg=cfg, backend=model.q_backend)
            _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Logged checkpoint at epoch {ep} (full dataset: {full_train_size} samples, best val_acc so far: {best_val_acc:.2f}%)")

    # Final: report and return best validation accuracy over all epochs (not just last epoch)
    _status_update({"stage": "train_done", "eval_id": eval_id,
                "final_val_loss": best_val_loss, "final_val_acc": best_val_acc})
    
    # Return best val loss/acc (highest accuracy from epoch 1 to end), for final_training.csv etc.
    return best_val_loss, best_val_acc, model, val_time, val_samples


"""Checkpoint validation training for correlation analysis."""
import time
from typing import List
from tqdm import tqdm

import torch

from ..utils.datasets import get_dataloaders
from ..utils.logging_utils import _append_progress, log_epoch, log_checkpoint, _get_worker_info
from ..models.config import QConfig
from ..utils.config import CHECKPOINT_VALIDATION_ENABLED


def _get_gpu_id():
    """Get current worker GPU ID."""
    gpu_id, _ = _get_worker_info()
    return gpu_id


def run_checkpoint_validation(model, loss_fn, cfg: QConfig, eval_id: str, 
                              checkpoint_sizes: List[int], in_pool_worker: bool,
                              full_train_size: int) -> None:
    """Run incremental training at checkpoint sizes, keeping weights between checkpoints.
    
    Trains incrementally: first on checkpoint_sizes[0] samples, then adds more data to reach
    checkpoint_sizes[1], and so on. The model keeps its weights between checkpoints, allowing
    study of how performance improves as training data increases.
    
    Args:
        model: The model to train incrementally (will be updated throughout)
        loss_fn: Loss function
        cfg: Model configuration
        eval_id: Evaluation ID for logging
        checkpoint_sizes: List of cumulative training data sizes (0 means full)
        in_pool_worker: Whether running in pool worker
        full_train_size: Size of the full training dataset
    """
    # Import evaluate here to avoid circular import
    from .trainer import evaluate
    
    if not CHECKPOINT_VALIDATION_ENABLED or not checkpoint_sizes:
        return
    
    _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Running incremental checkpoint training at sizes: {checkpoint_sizes}")
    
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    try:
        for idx, checkpoint_size in enumerate(checkpoint_sizes, 1):
            display_size = checkpoint_size if checkpoint_size > 0 else full_train_size
            _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Checkpoint {idx}/{len(checkpoint_sizes)}: "
                            f"Training on {display_size} samples (incremental)...")
            
            # Get dataloaders for this cumulative checkpoint size
            train_loader, val_loader = get_dataloaders(in_pool_worker=in_pool_worker, 
                                                       train_size=checkpoint_size if checkpoint_size > 0 else None,
                                                       val_size=None)
            
            # Train for 1 epoch at this checkpoint size (model keeps weights from previous checkpoint)
            model.train()
            run_loss, correct, seen = 0.0, 0, 0
            t0_epoch = time.time()
            
            # Gradient accumulation to reduce memory usage (especially important for CIFAR10)
            accumulation_steps = 2  # Effective batch size = BATCH_SIZE * accumulation_steps
            
            # Add progress bar for visibility
            total_batches = len(train_loader)
            pbar = tqdm(train_loader, desc=f"[GPU{_get_gpu_id()}|{eval_id}] Checkpoint {idx}/{len(checkpoint_sizes)}", leave=False)
            
            for batch_idx, (images, labels) in enumerate(pbar, 1):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # Only zero gradients at the start of accumulation
                    if (batch_idx - 1) % accumulation_steps == 0:
                        opt.zero_grad(set_to_none=True)
                    
                    out = model(images)
                    loss = loss_fn(out, labels)
                    
                    # Scale loss by accumulation steps
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    # Only step optimizer after accumulating gradients
                    if batch_idx % accumulation_steps == 0 or batch_idx == total_batches:
                        opt.step()
                    
                    bs = images.size(0)
                    run_loss += loss.item() * bs * accumulation_steps  # Unscale for logging
                    correct += (out.argmax(1) == labels).sum().item()
                    seen += bs
                    
                except KeyboardInterrupt:
                    # Handle interruption gracefully
                    _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Interrupted at checkpoint {idx}, batch {batch_idx}. Cleaning up...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # Re-raise to allow outer handler to clean up properly
                    raise
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] OOM at checkpoint {idx}, batch {batch_idx}. Clearing cache and continuing...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Skip this batch
                        continue
                    else:
                        # Re-raise if not OOM
                        raise
                
                # Update progress bar
                current_loss = run_loss / max(1, seen)
                current_acc = 100.0 * correct / max(1, seen)
                pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
                
                # Log batch progress to CSV (every N batches to avoid too much logging)
                # All checkpoint validations happen at epoch 1, so use epoch=1 for logging
                if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches:
                    log_epoch(eval_id, 1, current_loss, current_acc, None, None,
                             cfg, model.q_backend, phase="train_batch", batch_idx=batch_idx,
                             batches_total=total_batches, t0=t0_epoch)
                    
                    # Also log to progress file
                    _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Checkpoint {idx} batch {batch_idx}/{total_batches}: "
                                   f"loss={current_loss:.4f}, acc={current_acc:.2f}%")
                
                # Clean up
                del out, loss, images, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            tr_loss = run_loss / max(1, seen)
            tr_acc = 100.0 * correct / max(1, seen)
            
            # Evaluate on full validation set
            vloss, vacc, _, _ = evaluate(model, loss_fn, val_loader, cfg, max_batches=None, 
                                  epoch=1, tag=f"Checkpoint", eval_id=eval_id)
            
            # Log to train_epoch_log.csv - all checkpoint validations happen at epoch 1
            log_epoch(eval_id, 1, tr_loss, tr_acc, vloss, vacc, cfg, model.q_backend, 
                     phase="epoch_end", t0=t0_epoch)
            
            # Also log to checkpoint_validation.csv
            # For checkpoint validation, idx represents the checkpoint sequence (1, 2, 3...)
            # In the current implementation, all checkpoints happen at epoch 1
            log_checkpoint(eval_id, epoch=1, checkpoint_size=display_size, va_loss=vloss, 
                          va_acc=vacc, cfg=cfg, backend=model.q_backend)
            
            _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Checkpoint {idx} (train_size={display_size}): "
                            f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.2f}%, val_loss={vloss:.4f}, val_acc={vacc:.2f}%")
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        # Handle interruption at checkpoint level
        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] Checkpoint validation interrupted. Cleaning up GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Re-raise to allow outer handler to clean up properly
        raise


#!/usr/bin/env python3
"""
Continue training models from their last completed epoch to a target epoch.
Loads saved weights and continues training, appending results to existing CSV files.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Add project root to path (go up 3 levels from scripts/training/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after setting up path - use new modular structure
from qnas.models.hybrid_qnn import HybridQNN
from qnas.models.config import QConfig
from qnas.utils.datasets import get_dataloaders, IN_FEATURES, N_CLASSES
from qnas.utils.model_io import save_model_weights, load_model_weights
from qnas.utils.logging_utils import (
    log_epoch, log_checkpoint,
    _append_progress, _status_update,
    EPOCH_LOG_CSV, CHECKPOINT_LOG_CSV
)
from qnas.utils.config import (
    FINAL_SHOTS, DATASET,
    CHECKPOINT_TARGET_EPOCHS, CHECKPOINT_CORRELATION_ENABLED,
    BATCH_SIZE, FINAL_TRAIN_SUBSET_SIZE, FINAL_VAL_SUBSET_SIZE
)

# Parse CNOT modes string to list
def parse_cnot_modes(cnot_str: str) -> List[int]:
    """Convert CNOT mode string like 'even-all' to list of integers."""
    mode_map = {"all": 0, "odd": 1, "even": 2, "none": 3}
    parts = cnot_str.split("-")
    return [mode_map.get(p.lower(), 3) for p in parts]

# Parse ent_ranges string to list
def parse_ent_ranges(ent_str: str) -> List[int]:
    """Convert ent_ranges string like '1-2' to list of integers."""
    return [int(x) for x in ent_str.split("-")]

def load_model_config(weights_path: Path, checkpoint_csv: Path, train_epoch_csv: Path, eval_id: str) -> Optional[QConfig]:
    """
    Load model configuration, trying weights file first (if it has metadata), then CSV files.
    """
    # Try loading config from weights file first (if it has metadata)
    if weights_path.exists():
        try:
            state_dict, config_dict, metadata_dict = load_model_weights(str(weights_path))
            if config_dict is not None:
                # New format with metadata - create QConfig from it
                return QConfig(
                    embed_kind=config_dict['embed_kind'],
                    n_qubits=config_dict['n_qubits'],
                    depth=config_dict['depth'],
                    ent_ranges=config_dict['ent_ranges'],
                    cnot_modes=config_dict['cnot_modes'],
                    learning_rate=config_dict['learning_rate'],
                    shots=config_dict['shots']
                )
        except Exception as e:
            # If loading fails or old format, fall back to CSV
            pass
    
    # Fall back to CSV files
    return load_model_config_from_csv(checkpoint_csv, train_epoch_csv, eval_id)


def load_model_config_from_csv(checkpoint_csv: Path, train_epoch_csv: Path, eval_id: str) -> Optional[QConfig]:
    """Load model configuration from checkpoint_validation.csv or train_epoch_log.csv for a given eval_id."""
    # Try checkpoint_validation.csv first
    df = pd.read_csv(checkpoint_csv)
    model_rows = df[df['eval_id'] == eval_id]
    
    # If not found in checkpoint_validation.csv, try train_epoch_log.csv
    if model_rows.empty:
        # Handle malformed CSV rows
        import re
        import io
        with open(train_epoch_csv, 'r') as f:
            content = f.read()
        content_fixed = re.sub(r'(\d+\.\d+)(final-random)', r'\1\n\2', content)
        df = pd.read_csv(io.StringIO(content_fixed))
        # Filter to epoch_end entries to get config
        epoch_end_df = df[df['phase'] == 'epoch_end']
        model_rows = epoch_end_df[epoch_end_df['eval_id'] == eval_id]
        if model_rows.empty:
            raise ValueError(f"Model {eval_id} not found in either CSV file")
    
    # Get the first row for this eval_id (all rows should have same config)
    row = model_rows.iloc[0]
    
    embed = row['embed']
    n_qubits = int(row['n_qubits'])
    depth = int(row['depth'])
    ent_ranges = parse_ent_ranges(row['ent_ranges'])
    cnot_modes = parse_cnot_modes(row['cnot_modes'])
    lr = float(row['lr'])
    shots = FINAL_SHOTS  # Use final shots for continuation
    
    return QConfig(embed, n_qubits, depth, ent_ranges, cnot_modes, lr, shots)

def find_last_completed_epoch(df: pd.DataFrame, eval_id: str) -> int:
    """Find the last completed epoch for a given eval_id from epoch_end entries."""
    model_rows = df[df['eval_id'] == eval_id]
    if model_rows.empty:
        return 0
    # Get the max epoch from epoch_end entries (completed epochs)
    max_epoch = model_rows['epoch'].max()
    return int(max_epoch)

def find_models_needing_continuation(
    train_epoch_csv: Path,
    target_epoch: int
) -> List[Tuple[str, int]]:
    """
    Find models that need continuation.
    Reads from train_epoch_log.csv to get all completed epochs.
    Returns list of (eval_id, last_epoch) tuples.
    """
    # Read CSV file, handling malformed rows (some rows are missing newlines)
    import re
    with open(train_epoch_csv, 'r') as f:
        content = f.read()
    
    # Fix malformed rows: add newlines before 'final-random' patterns (but not at start)
    # Pattern: number followed by 'final-random' should have a newline inserted
    content_fixed = re.sub(r'(\d+\.\d+)(final-random)', r'\1\n\2', content)
    
    # Read the fixed content
    import io
    df = pd.read_csv(io.StringIO(content_fixed))
    
    # Filter to only epoch_end entries (completed epochs)
    epoch_end_df = df[df['phase'] == 'epoch_end'].copy()
    
    # Get all unique eval_ids from train_epoch_log.csv
    all_eval_ids = epoch_end_df['eval_id'].unique()
    
    models_to_continue = []
    for eval_id in all_eval_ids:
        last_epoch = find_last_completed_epoch(epoch_end_df, eval_id)
        if last_epoch < target_epoch:
            models_to_continue.append((eval_id, last_epoch))
    
    return sorted(models_to_continue)

def continue_training(
    run_dir: Path,
    eval_id: str,
    weights_path: Path,
    cfg: QConfig,
    start_epoch: int,
    target_epoch: int
):
    """Continue training a model from start_epoch to target_epoch."""
    
    # Set up environment for logging
    os.environ["QNAS_POOL_WORKER"] = "0"
    os.environ["IMPORTED_AS_MODULE"] = "true"
    os.environ["RUN_TYPE"] = "correlation"
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(
        in_pool_worker=False,
        train_size=FINAL_TRAIN_SUBSET_SIZE if FINAL_TRAIN_SUBSET_SIZE > 0 else None,
        val_size=FINAL_VAL_SUBSET_SIZE if FINAL_VAL_SUBSET_SIZE > 0 else None
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = HybridQNN(
        cfg.n_qubits, cfg.depth, cfg.ent_ranges, cfg.cnot_modes,
        cfg.embed_kind, cfg.shots, IN_FEATURES, N_CLASSES
    ).to_devices(device)
    
    # Load weights
    if not weights_path.exists():
        print(f"ERROR: Weights file not found: {weights_path}")
        return False
    
    state_dict, config_dict, metadata_dict = load_model_weights(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    
    if metadata_dict:
        print(f"Loaded weights from: {weights_path} (epoch {metadata_dict.get('epoch', 'unknown')}, eval_id: {metadata_dict.get('eval_id', 'unknown')})")
    else:
        print(f"Loaded weights from: {weights_path} (old format, no metadata)")
    
    # Create optimizer with same learning rate
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Continue training from start_epoch+1 to target_epoch
    # (start_epoch is already completed)
    actual_start = start_epoch + 1
    
    if actual_start > target_epoch:
        print(f"Model {eval_id} already at or past target epoch {target_epoch}")
        return True
    
    _append_progress(f"[{eval_id}] Continuing training from epoch {actual_start} to {target_epoch}")
    best_val_acc, best_val_loss = -1.0, None

    for ep in range(actual_start, target_epoch + 1):
        model.train()
        run_loss, correct, seen, it = 0.0, 0, 0, 0
        total_batches = len(train_loader)
        
        print(f"[{eval_id}] Training epoch {ep}/{target_epoch}...")
        
        for images, labels in train_loader:
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
                correct += (out.argmax(1) == labels).sum().item()
                seen += bs
                it += 1
                
                # Clean up intermediate tensors
                del out, loss, images, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "out of memory" in error_msg.lower():
                    print(f"ERROR: GPU memory error: {error_msg}")
                    return False
                else:
                    raise
        
        tr_loss = run_loss / max(1, seen)
        tr_acc = 100.0 * correct / max(1, seen)
        
        # Evaluate on validation set
        model.eval()
        val_loss, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                out = model(images)
                loss = loss_fn(out, labels)
                bs = images.size(0)
                val_loss += loss.item() * bs
                val_correct += (out.argmax(1) == labels).sum().item()
                val_seen += bs
        
        vloss = val_loss / max(1, val_seen)
        vacc = 100.0 * val_correct / max(1, val_seen)
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_val_loss = vloss
        if best_val_loss is None:
            best_val_loss = vloss

        # Log to train_epoch_log.csv
        log_epoch(eval_id, ep, tr_loss, tr_acc, vloss, vacc, cfg, model.q_backend, phase="epoch_end")

        # Log to checkpoint_validation.csv if this epoch is in CHECKPOINT_TARGET_EPOCHS (best val_acc so far)
        if CHECKPOINT_CORRELATION_ENABLED and ep in CHECKPOINT_TARGET_EPOCHS:
            full_train_dataset, _ = get_dataloaders(in_pool_worker=False, train_size=None, val_size=None)
            full_train_size = len(full_train_dataset.dataset) if hasattr(full_train_dataset, 'dataset') else 0
            log_checkpoint(eval_id, epoch=ep, checkpoint_size=full_train_size, va_loss=best_val_loss,
                          va_acc=best_val_acc, cfg=cfg, backend=model.q_backend)
        
        print(f"[{eval_id}] Ep {ep}/{target_epoch} | train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | val_loss={vloss:.4f} acc={vacc:.2f}%")
    
    # Save updated weights
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    updated_weights_path = weights_dir / f"{eval_id.replace('final-', '')}_final.pt"
    save_model_weights(model, str(updated_weights_path), cfg, eval_id=eval_id, epoch=target_epoch, val_acc=vacc, val_loss=vloss)
    print(f"Saved updated weights to: {updated_weights_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Continue training models from their last completed epoch to a target epoch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continue to epoch 10 (default, from CHECKPOINT_TARGET_EPOCHS)
  python continue_training.py logs/correlation/MNIST/run_20251128-171506
  
  # Continue to a specific epoch
  python continue_training.py logs/correlation/MNIST/run_20251128-171506 --target-epoch 10
  
  # Continue only specific models
  python continue_training.py logs/correlation/MNIST/run_20251128-171506 --eval-ids final-random-0001 final-random-0002
        """
    )
    parser.add_argument("run_dir", type=Path, help="Path to the run directory containing train_epoch_log.csv, checkpoint_validation.csv and weights/")
    parser.add_argument("--target-epoch", type=int, default=None,
                       help="Target epoch to train to (default: max from CHECKPOINT_TARGET_EPOCHS or 10)")
    parser.add_argument("--eval-ids", nargs="+", default=None,
                       help="Specific eval_ids to continue (default: all models needing continuation)")
    
    args = parser.parse_args()
    
    run_dir = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        sys.exit(1)
    
    train_epoch_csv = run_dir / "train_epoch_log.csv"
    if not train_epoch_csv.exists():
        print(f"ERROR: train_epoch_log.csv not found in {run_dir}")
        sys.exit(1)
    
    checkpoint_csv = run_dir / "checkpoint_validation.csv"
    if not checkpoint_csv.exists():
        print(f"ERROR: checkpoint_validation.csv not found in {run_dir}")
        sys.exit(1)
    
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        print(f"ERROR: weights directory not found in {run_dir}")
        sys.exit(1)
    
    # Determine target epoch
    if args.target_epoch is not None:
        target_epoch = args.target_epoch
    else:
        # Use max from CHECKPOINT_TARGET_EPOCHS, or default to 10
        if CHECKPOINT_TARGET_EPOCHS:
            target_epoch = max(CHECKPOINT_TARGET_EPOCHS)
        else:
            target_epoch = 10
    
    print(f"Target epoch: {target_epoch}")
    
    # Find models that need continuation (read from train_epoch_log.csv for accurate epoch tracking)
    models_to_continue = find_models_needing_continuation(train_epoch_csv, target_epoch)
    
    # Filter to specific eval_ids if requested
    if args.eval_ids:
        requested_set = set(args.eval_ids)
        models_to_continue = [(eid, last) for eid, last in models_to_continue if eid in requested_set]
        if not models_to_continue:
            print(f"ERROR: None of the specified eval_ids need continuation to epoch {target_epoch}")
            sys.exit(1)
    
    if not models_to_continue:
        print(f"No models need continuation (all models already at or past epoch {target_epoch})")
        sys.exit(0)
    
    print(f"\nFound {len(models_to_continue)} models that need continuation:")
    for eid, last_epoch in models_to_continue:
        print(f"  - {eid}: last epoch {last_epoch} -> target epoch {target_epoch}")
    
    # Set up logging paths by setting environment variables and module attributes
    os.environ["LOG_DIR"] = str(run_dir)
    os.environ["DATASET_LOG_DIR"] = str(run_dir)
    
    # Import and update module-level attributes
    import qnas.main as qnas_main
    import qnas.utils.logging_utils as lu
    import qnas.utils.config as cfg
    
    qnas_main.LOG_DIR = str(run_dir)
    qnas_main.DATASET_LOG_DIR = str(run_dir)
    lu.DATASET_LOG_DIR = str(run_dir)
    cfg.LOG_DIR = str(run_dir)
    cfg.DATASET_LOG_DIR = str(run_dir)
    
    # Update CSV paths in logging_utils
    lu.EPOCH_LOG_CSV = os.path.join(str(run_dir), "train_epoch_log.csv")
    lu.CHECKPOINT_LOG_CSV = os.path.join(str(run_dir), "checkpoint_validation.csv")
    lu.PROGRESS_LOG = os.path.join(str(run_dir), "progress.log")
    
    # Process each model
    success_count = 0
    for eval_id, last_epoch in models_to_continue:
        print(f"\n{'='*60}")
        print(f"Processing: {eval_id} (last epoch: {last_epoch}, target: {target_epoch})")
        print(f"{'='*60}")
        
        # Find weights file (remove 'final-' prefix if present)
        weight_id = eval_id.replace('final-', '')
        weights_path = weights_dir / f"{weight_id}_final.pt"
        
        if not weights_path.exists():
            print(f"WARNING: Weights file not found: {weights_path}, skipping...")
            continue
        
        # Load config (tries weights file first, then CSV files)
        try:
            model_cfg = load_model_config(weights_path, checkpoint_csv, train_epoch_csv, eval_id)
        except Exception as e:
            print(f"ERROR: Failed to load config for {eval_id}: {e}")
            continue
        
        # Continue training
        try:
            success = continue_training(
                run_dir, eval_id, weights_path, model_cfg,
                start_epoch=last_epoch, target_epoch=target_epoch
            )
            if success:
                success_count += 1
                print(f"✓ Successfully continued training for {eval_id}")
            else:
                print(f"✗ Failed to continue training for {eval_id}")
        except Exception as e:
            print(f"✗ Error continuing training for {eval_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(models_to_continue)} models successfully continued")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


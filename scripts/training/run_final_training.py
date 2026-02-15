#!/usr/bin/env python3
"""
Run final training on Pareto optimal configurations from NSGA-II results.
Trains all Pareto optimal points using both available GPUs.
"""

import os
import sys
import csv
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path (go up 3 levels from scripts/training/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_log_directory(csv_path: Path):
    """Extract run directory from CSV path and set environment variables.
    
    MUST be called BEFORE importing any src modules to prevent creating new run folders.
    """
    # Get the run directory (parent of the CSV file)
    run_dir = csv_path.parent.resolve()
    
    # Set environment variables BEFORE importing modules
    # This prevents config.py from creating a new run folder
    os.environ["DATASET_LOG_DIR"] = str(run_dir)
    os.environ["LOG_DIR"] = str(run_dir.parent.parent)  # logs/nsga-ii/MNIST
    os.environ["IMPORTED_AS_MODULE"] = "false"  # But we're running, not importing
    os.environ["RUN_TYPE"] = "nsga"  # We're continuing NSGA-II run
    
    # Set CSV paths to use existing files
    os.environ["NSGA_EVAL_CSV"] = str(csv_path)
    os.environ["EPOCH_LOG_CSV"] = str(run_dir / "train_epoch_log.csv")
    os.environ["GEN_SUMMARY_CSV"] = str(run_dir / "nsga_gen_summary.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(run_dir / "checkpoint_validation.csv")
    os.environ["PROGRESS_LOG"] = str(run_dir / "progress.log")
    
    print(f"Using existing run directory: {run_dir}")
    return run_dir

def parse_cnot_modes(cnot_str: str) -> list:
    """Convert CNOT mode string like 'none-even' to list of integers."""
    mode_map = {"all": 0, "odd": 1, "even": 2, "none": 3}
    parts = cnot_str.split("-")
    return [mode_map.get(p.lower(), 3) for p in parts]

def parse_ent_ranges(ent_str: str) -> list:
    """Convert ent_ranges string like '2-4' to list of integers."""
    return [int(x) for x in ent_str.split("-")]

# CSV header for final training results
FINAL_TRAINING_HEADER = [
    "eval_id", "original_eval_id", "embed_kind", "n_qubits", "depth", 
    "ent_ranges", "cnot_modes", "learning_rate", "shots",
    "nsga_val_acc", "nsga_f2_circuit_cost", "final_val_acc", "final_val_loss",
    "gpu_id", "success", "save_path", "error"
]

def find_pareto_front(data, obj_cols):
    """Find Pareto optimal points (minimization for all objectives).
    
    Args:
        data: DataFrame with evaluation results
        obj_cols: List of column names for objectives to minimize
        
    Returns:
        Array of indices for Pareto optimal points
    """
    costs = data[obj_cols].values
    n_points = costs.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            # Check if point j dominates point i
            # j dominates i if: j is <= i in all objectives AND j < i in at least one
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_pareto[i] = False
                break
    
    return np.where(is_pareto)[0]

def train_single_config(config_dict: dict, gpu_id: int):
    """Train a single configuration on a specific GPU.
    
    Args:
        config_dict: Dictionary containing configuration and metadata
        gpu_id: GPU ID to use for training
        
    Returns:
        Dictionary with training results
    """
    # Set GPU BEFORE importing torch/src modules
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["QNAS_POOL_WORKER"] = "0"
    os.environ["FINAL_TRAIN_GPU"] = str(gpu_id)
    
    # Set log directory environment variables (critical for spawn method)
    run_dir = Path(config_dict['run_dir'])
    os.environ["DATASET_LOG_DIR"] = str(run_dir)
    os.environ["LOG_DIR"] = str(run_dir.parent.parent)
    os.environ["IMPORTED_AS_MODULE"] = "false"
    os.environ["RUN_TYPE"] = "nsga"
    os.environ["NSGA_EVAL_CSV"] = str(run_dir / "nsga_evals.csv")
    os.environ["EPOCH_LOG_CSV"] = str(run_dir / "train_epoch_log.csv")
    os.environ["GEN_SUMMARY_CSV"] = str(run_dir / "nsga_gen_summary.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(run_dir / "checkpoint_validation.csv")
    os.environ["PROGRESS_LOG"] = str(run_dir / "progress.log")
    
    # Now import modules - use new modular structure
    import torch
    from qnas.models.config import QConfig
    from qnas.training.trainer import train_for_budget
    from qnas.utils.model_io import save_model_weights
    from qnas.utils.config import FINAL_SHOTS, FINAL_TRAIN_EPOCHS, FINAL_TRAIN_SUBSET_SIZE, FINAL_VAL_SUBSET_SIZE
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Only one GPU visible
    
    eval_id = config_dict['eval_id']
    original_id = config_dict.get('original_eval_id', eval_id)
    print(f"\n[GPU {gpu_id}] Training {eval_id} (original: {original_id})...")
    print(f"[GPU {gpu_id}]   Accuracy: {config_dict['val_acc']:.2f}%")
    print(f"[GPU {gpu_id}]   Circuit cost: {config_dict['f2_circuit_cost']:.6f}s/sample")
    
    # Create QConfig
    cfg = QConfig(
        embed_kind=config_dict['embed_kind'],
        n_qubits=config_dict['n_qubits'],
        depth=config_dict['depth'],
        ent_ranges=config_dict['ent_ranges'],
        cnot_modes=config_dict['cnot_modes'],
        learning_rate=config_dict['learning_rate'],
        shots=FINAL_SHOTS
    )
    
    try:
        # Train
        vloss, vacc, model, _, _ = train_for_budget(
            cfg, eval_id, FINAL_TRAIN_EPOCHS, 0, 0,
            train_size=FINAL_TRAIN_SUBSET_SIZE, 
            val_size=FINAL_VAL_SUBSET_SIZE
        )
        
        # Save weights to weights/{run_folder}/
        run_folder_name = run_dir.name
        weights_dir = Path("weights") / run_folder_name
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_path = weights_dir / f"hybrid_qnn_pareto_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}_{eval_id}.pt"
        save_model_weights(model, str(save_path), cfg, eval_id=eval_id, epoch=FINAL_TRAIN_EPOCHS, val_acc=vacc, val_loss=vloss)
        
        print(f"[GPU {gpu_id}] ✓ {eval_id} - Final acc: {vacc:.2f}%, loss: {vloss:.4f}")
        print(f"[GPU {gpu_id}]   Saved: {save_path}")
        
        return {
            'eval_id': eval_id,
            'original_eval_id': original_id,
            'gpu_id': gpu_id,
            'success': True,
            'val_acc': vacc,
            'val_loss': vloss,
            'save_path': str(save_path),
            'embed_kind': cfg.embed_kind,
            'n_qubits': cfg.n_qubits,
            'depth': cfg.depth,
            'ent_ranges': cfg.ent_ranges,
            'cnot_modes': cfg.cnot_modes,
            'learning_rate': cfg.learning_rate,
            'shots': cfg.shots
        }
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ {eval_id} - Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'eval_id': eval_id,
            'original_eval_id': original_id,
            'gpu_id': gpu_id,
            'success': False,
            'error': str(e),
            'embed_kind': config_dict.get('embed_kind', ''),
            'n_qubits': config_dict.get('n_qubits', ''),
            'depth': config_dict.get('depth', ''),
            'ent_ranges': config_dict.get('ent_ranges', []),
            'cnot_modes': config_dict.get('cnot_modes', []),
            'learning_rate': config_dict.get('learning_rate', ''),
            'shots': FINAL_SHOTS
        }

def main():
    import argparse
    import pandas as pd
    import torch
    import multiprocessing as mp
    
    # CRITICAL: Set spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Run final training on Pareto optimal NSGA-II configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all Pareto optimal configs from a specific run
  python scripts/run_final_training.py logs/nsga-ii/MNIST/run_20251227-003334/nsga_evals.csv
  
  # Train with specific GPUs
  python scripts/run_final_training.py logs/nsga-ii/MNIST/run_20251227-003334/nsga_evals.csv --gpus 0 1
        """
    )
    parser.add_argument("csv_path", type=Path, 
                       help="Path to nsga_evals.csv file")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1],
                       help="GPU IDs to use (default: 0 1)")
    parser.add_argument("--objectives", type=str, nargs="+", 
                       default=["f1_1_minus_acc", "f2_circuit_cost"],
                       help="Objectives for Pareto front (default: f1_1_minus_acc f2_circuit_cost)")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Check GPU availability
    available_gpus = []
    if torch.cuda.is_available():
        for gpu_id in args.gpus:
            if gpu_id < torch.cuda.device_count():
                available_gpus.append(gpu_id)
            else:
                print(f"Warning: GPU {gpu_id} not available")
    
    if not available_gpus:
        print("ERROR: No GPUs available")
        sys.exit(1)
    
    print(f"Using GPUs: {available_gpus}")
    
    # CRITICAL: Set log directory BEFORE importing any src modules
    run_dir = setup_log_directory(csv_path)
    print()
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} evaluations from {csv_path}")
    
    # Find Pareto optimal points
    print(f"\nFinding Pareto optimal points using objectives: {args.objectives}")
    pareto_indices = find_pareto_front(df, args.objectives)
    pareto_df = df.iloc[pareto_indices].sort_values('val_acc', ascending=False)
    
    print(f"\nFound {len(pareto_df)} Pareto optimal configurations:")
    print("="*80)
    for idx, row in pareto_df.iterrows():
        print(f"  {row['eval_id']}:")
        print(f"    Accuracy: {row['val_acc']:.2f}%")
        print(f"    F1 (1-acc): {row['f1_1_minus_acc']:.6f}")
        print(f"    F2 (circuit cost): {row['f2_circuit_cost']:.6f} s/sample")
        if 'f3_n_subcircuits' in row:
            print(f"    F3 (subcircuits): {int(row['f3_n_subcircuits'])}")
        print(f"    Config: {row['embed']}, {int(row['n_qubits'])}q, depth={int(row['depth'])}")
        print()
    
    # Prepare configurations with descriptive names
    configs = []
    
    # Sort by accuracy to identify best/worst
    pareto_sorted_acc = pareto_df.sort_values('val_acc', ascending=False)
    pareto_sorted_cost = pareto_df.sort_values('f2_circuit_cost', ascending=True)
    
    best_acc_id = pareto_sorted_acc.iloc[0]['eval_id']
    lowest_cost_id = pareto_sorted_cost.iloc[0]['eval_id']
    
    # Counter for balanced configurations (not using iteration index)
    balanced_counter = 1
    
    for i, (idx, row) in enumerate(pareto_df.iterrows()):
        # Assign descriptive name based on characteristics
        if row['eval_id'] == best_acc_id:
            pareto_name = "final-pareto-best_accuracy"
        elif row['eval_id'] == lowest_cost_id:
            pareto_name = "final-pareto-lowest_cost"
        else:
            pareto_name = f"final-pareto-balanced_{balanced_counter}"
            balanced_counter += 1
        
        configs.append({
            'eval_id': pareto_name,
            'original_eval_id': row['eval_id'],
            'embed_kind': row['embed'],
            'n_qubits': int(row['n_qubits']),
            'depth': int(row['depth']),
            'ent_ranges': parse_ent_ranges(str(row['ent_ranges'])),
            'cnot_modes': parse_cnot_modes(str(row['cnot_modes'])),
            'learning_rate': float(row['learning_rate']),
            'val_acc': row['val_acc'],
            'f2_circuit_cost': row['f2_circuit_cost'],
            'run_dir': str(run_dir)  # Pass run_dir to each config
        })
    
    print("="*80)
    print(f"\nStarting parallel training on {len(available_gpus)} GPUs...")
    print(f"Results will be saved to: {run_dir}/weights/")
    
    # Create final_training.csv file in the run directory (use absolute path)
    final_training_csv = run_dir.resolve() / "final_training.csv"
    print(f"Final training results will be logged to: {final_training_csv}")
    print(f"  (absolute path: {final_training_csv.absolute()})")
    print()
    
    # Initialize CSV file with header
    try:
        with open(final_training_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_TRAINING_HEADER)
            writer.writeheader()
        print(f"[CSV] Initialized {final_training_csv} with header")
    except Exception as e:
        print(f"[CSV] ERROR initializing {final_training_csv}: {e}")
        raise
    
    # Train in parallel using multiple GPUs
    results = []
    with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = {}
        for i, config in enumerate(configs):
            gpu_id = available_gpus[i % len(available_gpus)]
            future = executor.submit(train_single_config, config, gpu_id)
            futures[future] = (config['eval_id'], gpu_id)
        
        # Collect results as they complete and write to CSV
        for future in as_completed(futures):
            eval_id, gpu_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                # Find the original config to get NSGA-II metrics
                config = next((c for c in configs if c['eval_id'] == eval_id), None)
                
                # Prepare CSV row
                row = {
                    'eval_id': result['eval_id'],
                    'original_eval_id': result.get('original_eval_id', ''),
                    'embed_kind': result.get('embed_kind', ''),
                    'n_qubits': result.get('n_qubits', ''),
                    'depth': result.get('depth', ''),
                    'ent_ranges': '-'.join(map(str, result.get('ent_ranges', []))),
                    'cnot_modes': '-'.join(map(str, result.get('cnot_modes', []))),
                    'learning_rate': f"{result.get('learning_rate', 0):.6e}" if result.get('learning_rate') is not None and result.get('learning_rate') != '' else '',
                    'shots': str(result.get('shots', '')) if result.get('shots') is not None else '',
                    'nsga_val_acc': f"{config['val_acc']:.4f}" if config else '',
                    'nsga_f2_circuit_cost': f"{config['f2_circuit_cost']:.6f}" if config else '',
                    'final_val_acc': f"{result.get('val_acc', ''):.4f}" if result.get('success') else '',
                    'final_val_loss': f"{result.get('val_loss', ''):.6f}" if result.get('success') else '',
                    'gpu_id': result['gpu_id'],
                    'success': 'True' if result.get('success') else 'False',
                    'save_path': result.get('save_path', ''),
                    'error': result.get('error', '')
                }
                
                # Append to CSV
                try:
                    with open(final_training_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=FINAL_TRAINING_HEADER)
                        writer.writerow(row)
                    print(f"[CSV] Wrote result for {eval_id} to {final_training_csv}")
                except Exception as csv_err:
                    print(f"[CSV] ERROR writing to {final_training_csv}: {csv_err}")
                    
            except Exception as e:
                print(f"\n✗ Error training {eval_id} on GPU {gpu_id}: {e}")
                import traceback
                traceback.print_exc()
                
                # Write error to CSV
                config = next((c for c in configs if c['eval_id'] == eval_id), None)
                row = {
                    'eval_id': eval_id,
                    'original_eval_id': config.get('original_eval_id', '') if config else '',
                    'embed_kind': config.get('embed_kind', '') if config else '',
                    'n_qubits': config.get('n_qubits', '') if config else '',
                    'depth': config.get('depth', '') if config else '',
                    'ent_ranges': '-'.join(map(str, config.get('ent_ranges', []))) if config else '',
                    'cnot_modes': '-'.join(map(str, config.get('cnot_modes', []))) if config else '',
                    'learning_rate': f"{config.get('learning_rate', 0):.6e}" if config and config.get('learning_rate') is not None else '',
                    'shots': '',
                    'nsga_val_acc': f"{config['val_acc']:.4f}" if config else '',
                    'nsga_f2_circuit_cost': f"{config['f2_circuit_cost']:.6f}" if config else '',
                    'final_val_acc': '',
                    'final_val_loss': '',
                    'gpu_id': gpu_id,
                    'success': 'False',
                    'save_path': '',
                    'error': str(e)
                }
                try:
                    with open(final_training_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=FINAL_TRAINING_HEADER)
                        writer.writerow(row)
                    print(f"[CSV] Wrote error for {eval_id} to {final_training_csv}")
                except Exception as csv_err:
                    print(f"[CSV] ERROR writing error to {final_training_csv}: {csv_err}")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        print(f"\n✓ Successfully trained {len(successful)}/{len(configs)} configurations:")
        for r in sorted(successful, key=lambda x: x['val_acc'], reverse=True):
            print(f"  {r['eval_id']}: acc={r['val_acc']:.2f}%, loss={r['val_loss']:.4f} [GPU {r['gpu_id']}]")
    
    if failed:
        print(f"\n✗ Failed to train {len(failed)} configurations:")
        for r in failed:
            print(f"  {r['eval_id']}: {r['error']} [GPU {r['gpu_id']}]")
    
    print(f"\nFinal training results saved to: {final_training_csv}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

"""NSGA-II execution and final training for Hybrid Quantum Neural Networks."""
import os
import sys
import csv
import time
import json
import threading
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import StarmapParallelization

from ..models.config import QConfig
from ..quantum.metrics import _modes_str
from ..utils import logging_utils as log_utils
from ..utils.model_io import save_model_weights
from ..utils.config import (
    POP_SIZE, N_GEN, SEED, BATCH_SIZE, SHOTS, EVAL_EPOCHS,
    RESUME_LOGS, WORKERS_PER_GPU,
    FINAL_TRAIN_GPU, FINAL_TRAIN_GPUS, FINAL_TRAIN_EPOCHS, FINAL_SHOTS, FINAL_WORKERS_PER_GPU,
    FINAL_TRAIN_SUBSET_SIZE, FINAL_VAL_SUBSET_SIZE,
    PARETO_OBJECTIVES,
    WORKER_GPU_ID as DEFAULT_WORKER_GPU_ID,
    WORKER_RANK as DEFAULT_WORKER_RANK,
    STATUS_JSON_PATH as DEFAULT_STATUS_JSON_PATH
)
from ..utils import config as cfg
from .problem import QNNHyperProblem, CSV_LOCK, GLOBAL_COUNTER, CURRENT_GENERATION

# Module-level variables for worker state
WORKER_RANK = DEFAULT_WORKER_RANK
WORKER_GPU_ID = DEFAULT_WORKER_GPU_ID
STATUS_JSON_PATH = DEFAULT_STATUS_JSON_PATH

# CSV header for final training results
FINAL_TRAINING_HEADER = [
    "eval_id", "original_eval_id", "embed_kind", "n_qubits", "depth", 
    "ent_ranges", "cnot_modes", "learning_rate", "shots",
    "nsga_val_acc", "nsga_f2_circuit_cost", "final_val_acc", "final_val_loss",
    "gpu_id", "success", "save_path", "error"
]


def _mp_init(gpu_ids: List[int], log_dir: str, seed_base: int, lock_proxy, counter_proxy, generation_proxy, workers_per_gpu: int):
    """Initializer for each Pool worker process; pins the worker to a specific GPU."""
    global WORKER_RANK, WORKER_GPU_ID, STATUS_JSON_PATH
    from . import problem as prob_module
    
    prob_module.CSV_LOCK = lock_proxy
    prob_module.GLOBAL_COUNTER = counter_proxy
    prob_module.CURRENT_GENERATION = generation_proxy

    proc = mp.current_process()
    local_rank = (proc._identity[0] - 1) if proc._identity else 0
    WORKER_RANK = local_rank
    # Distribute workers across GPUs based on WORKERS_PER_GPU setting
    gpu_index = (local_rank // workers_per_gpu) % len(gpu_ids)
    WORKER_GPU_ID = int(gpu_ids[gpu_index])

    # Update config module's worker variables
    from ..utils.config import _update_worker_info
    _update_worker_info(WORKER_GPU_ID, WORKER_RANK)

    # Mark this as a pool worker so DataLoader enforces num_workers=0
    os.environ["QNAS_POOL_WORKER"] = "1"

    # Pin this process to exactly one GPU (make it appear as device 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(WORKER_GPU_ID)

    # Silence TF/absl logs *if* TensorFlow gets imported later by cutter;
    # DO NOT import TensorFlow here (that import is what triggers the spam).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide INFO/WARN
    os.environ.setdefault("GLOG_minloglevel", "3")       # XLA/glog (some TF builds)
    os.environ.setdefault("ABSL_LOG_SEVERITY", "FATAL")  # absl pre-init logs

    # Keep PyTorch bound to our single visible device in this worker
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    except Exception:
        pass

    torch.manual_seed(seed_base + local_rank)
    np.random.seed(seed_base + local_rank)

    log_utils.refresh_logging_paths()
    STATUS_JSON_PATH = os.path.join(log_utils.STATUS_DIR, f"worker_gpu{WORKER_GPU_ID}_status.json")


def _start_status_watcher(stop_evt: threading.Event, gpu_ids: List[int]):
    """Print a compact per-GPU status line every ~2s while NSGA runs."""
    def run():
        last = {}
        while not stop_evt.is_set():
            time.sleep(2.0)
            lines = []
            for gid in gpu_ids:
                path = os.path.join(log_utils.STATUS_DIR, f"worker_gpu{gid}_status.json")
                try:
                    with open(path, "r") as f:
                        s = json.load(f)
                except Exception:
                    s = None
                if not s:
                    lines.append(f"[WATCH] GPU{gid}: idle")
                    continue
                stage = s.get("stage", "?")
                eid = s.get("eval_id", "-")
                lines.append(f"[WATCH] GPU{gid}: {stage} {eid}")
            msg = "   ".join(lines)
            # Print only when changed to reduce noise
            if msg != last.get("msg"):
                print(msg, flush=True)
                last["msg"] = msg
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def run_nsga2() -> QConfig:
    """Run NSGA-II optimization to find optimal quantum circuit configurations.
    
    Returns:
        QConfig: Best configuration found by NSGA-II
    """
    # Use spawn for CUDA safety (pool & DL workers)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Mark main process (used to suppress import-time prints elsewhere)
    os.environ["QNAS_MAIN"] = "1"

    # Create (or reuse) run directory explicitly at runtime.
    run_dir = cfg.initialize_nsga_run_dir(force_new=False, copy_env_snapshot=True)
    os.environ["DATASET_LOG_DIR"] = run_dir
    os.environ["NSGA_EVAL_CSV"] = str(Path(run_dir) / "nsga_evals.csv")
    os.environ["EPOCH_LOG_CSV"] = str(Path(run_dir) / "train_epoch_log.csv")
    os.environ["GEN_SUMMARY_CSV"] = str(Path(run_dir) / "nsga_gen_summary.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(Path(run_dir) / "checkpoint_validation.csv")
    os.environ["PROGRESS_LOG"] = str(Path(run_dir) / "progress.log")
    log_utils.refresh_logging_paths(run_dir)

    # Reset logs at the very beginning of a run, unless RESUME_LOGS=1
    if not RESUME_LOGS:
        log_utils._csv_reset_all()

    # So users always know where to find logs (DATASET_LOG_DIR is the run folder)
    run_dir_abs = os.path.abspath(cfg.DATASET_LOG_DIR) if cfg.DATASET_LOG_DIR else "(none)"
    print(f"Logs: {run_dir_abs}")

    num_visible = torch.cuda.device_count()
    gpu_ids = list(range(num_visible)) if num_visible > 0 else []
    print(f"Detected GPUs: {gpu_ids if gpu_ids else 'CPU only'}")
    log_utils._append_progress(
        f"[MAIN] Using GPUs {gpu_ids if gpu_ids else 'CPU only'} | "
        f"BATCH_SIZE={BATCH_SIZE}, epochs per eval={EVAL_EPOCHS}, shots={SHOTS if SHOTS else 'adjoint'}"
    )

    manager = mp.Manager()
    lock_proxy = manager.RLock()
    counter_proxy = manager.Value('i', 0)
    generation_proxy = manager.Value('i', 0)
    
    # Update problem module's shared variables
    from . import problem as prob_module
    prob_module.CSV_LOCK = lock_proxy
    prob_module.GLOBAL_COUNTER = counter_proxy
    prob_module.CURRENT_GENERATION = generation_proxy

    pool_size = max(1, len(gpu_ids) * WORKERS_PER_GPU)  # configurable workers per GPU
    
    pool = mp.Pool(processes=pool_size,
                   initializer=_mp_init,
                   initargs=(gpu_ids if gpu_ids else [-1], cfg.LOG_DIR, SEED, lock_proxy, counter_proxy, generation_proxy, WORKERS_PER_GPU))

    # Live status watcher in main process
    stop_evt = threading.Event()
    watcher = _start_status_watcher(stop_evt, gpu_ids if gpu_ids else [-1])

    runner = StarmapParallelization(pool.starmap)
    problem = QNNHyperProblem(elementwise_runner=runner)
    algorithm = NSGA2(pop_size=POP_SIZE)
    termination = get_termination("n_gen", N_GEN)

    from ..utils.config import CUT_TARGET_QUBITS
    n_obj = problem.n_obj
    obj_desc = f"f1, f2" if n_obj == 2 else f"f1, f2, f3"
    cut_status = "disabled (CUT_TARGET_QUBITS=0)" if CUT_TARGET_QUBITS <= 0 else f"enabled (CUT_TARGET_QUBITS={CUT_TARGET_QUBITS})"
    print(f"NSGA-II starting | pop={POP_SIZE}, gens={N_GEN}, pool_size={pool_size} ({WORKERS_PER_GPU} workers/GPU)")
    print(f"  Objectives: {obj_desc} | Wire cutting: {cut_status}")
    # Debug: Show what was actually loaded
    env_val = os.environ.get('CUT_TARGET_QUBITS', 'NOT SET')
    print(f"  [DEBUG] CUT_TARGET_QUBITS from os.environ: {env_val}, from config: {CUT_TARGET_QUBITS}, n_obj: {n_obj}")
    run_t0 = time.time()
    
    # Initialize generation counter to 0 before algorithm starts
    # This ensures evaluations in generation 0 can read the correct value
    from . import problem as prob_module
    if prob_module.CURRENT_GENERATION is not None:
        try:
            with prob_module.CSV_LOCK:
                prob_module.CURRENT_GENERATION.value = 0
        except Exception:
            log_utils._append_progress("[WARN] Could not initialize generation counter; defaulting to 0")

    def _shutdown_pool(terminate: bool = False):
        stop_evt.set()
        try:
            if terminate:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        except Exception:
            pass
        try:
            manager.shutdown()
        except Exception:
            pass

    from .callbacks import ProgressCallback
    cb = ProgressCallback(generation_proxy=generation_proxy, csv_lock=lock_proxy)
    try:
        res = minimize(problem, algorithm, termination, seed=SEED, save_history=False, verbose=False, callback=cb)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted. Shutting down workers...")
        _shutdown_pool(terminate=True)
        print("Done. Logs saved to:", run_dir_abs)
        sys.exit(130)

    # Stop watcher & close workers (graceful)
    _shutdown_pool(terminate=False)

    # Sanity: show remaining active children (should be 0 here)
    rem = mp.active_children()
    print(f"[MAIN] NSGA pool joined. Active children now: {len(rem)}")
    for ch in rem:
        print(f" - {ch.name} (PID={ch.pid})")

    F, X = res.F, res.X
    best_idx = int(np.argmin(F[:,0]))
    best_cfg = problem._decode(X[best_idx])

    print("\n=== NSGA-II Done ===")
    # Handle variable number of objectives (f3 only present when CUT_TARGET_QUBITS > 0)
    n_sub_str = f" | n_sub={int(F[best_idx,2])}" if F.shape[1] > 2 else ""
    print(f"Best (by accuracy): acc≈{(1-F[best_idx,0])*100:.2f}% | time_cost={F[best_idx,1]:.6f}s/sample{n_sub_str}")
    print(f"  embed={best_cfg.embed_kind}, n_qubits={best_cfg.n_qubits}, depth={best_cfg.depth}, "
          f"ranges={best_cfg.ent_ranges}, cnot={_modes_str(best_cfg.cnot_modes)}, lr={best_cfg.learning_rate:.2e}")

    return best_cfg


def _parse_cnot_modes(cnot_str: str) -> list:
    """Convert CNOT mode string like 'none-even' to list of integers."""
    mode_map = {"all": 0, "odd": 1, "even": 2, "none": 3}
    parts = cnot_str.split("-")
    return [mode_map.get(p.lower(), 3) for p in parts]


def _parse_ent_ranges(ent_str: str) -> list:
    """Convert ent_ranges string like '2-4' to list of integers."""
    return [int(x) for x in ent_str.split("-")]


def _find_pareto_front(data: pd.DataFrame, obj_cols: List[str]) -> np.ndarray:
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


def _train_single_config(config_dict: dict, gpu_id: int) -> dict:
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
    logs_root = run_dir.parents[2] if len(run_dir.parents) >= 3 else run_dir.parent
    os.environ["DATASET_LOG_DIR"] = str(run_dir)
    os.environ["LOG_DIR"] = str(logs_root)
    os.environ["IMPORTED_AS_MODULE"] = "false"
    os.environ["RUN_TYPE"] = "nsga"
    os.environ["NSGA_EVAL_CSV"] = str(run_dir / "nsga_evals.csv")
    os.environ["EPOCH_LOG_CSV"] = str(run_dir / "train_epoch_log.csv")
    os.environ["GEN_SUMMARY_CSV"] = str(run_dir / "nsga_gen_summary.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(run_dir / "checkpoint_validation.csv")
    os.environ["PROGRESS_LOG"] = str(run_dir / "progress.log")
    
    from ..utils import config as cfg_module
    from ..utils import logging_utils as lu
    cfg_module.set_dataset_log_dir(str(run_dir), create=False)
    lu.refresh_logging_paths(str(run_dir))

    # Now import modules
    import torch
    from ..models.config import QConfig
    from ..training.trainer import train_for_budget
    from ..utils.model_io import save_model_weights
    from ..utils.config import FINAL_SHOTS, FINAL_TRAIN_EPOCHS, FINAL_TRAIN_SUBSET_SIZE, FINAL_VAL_SUBSET_SIZE
    
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
        
        # Save weights to weights/{run_folder}/ instead of logs/{run_folder}/weights/
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


def final_train(best_cfg: QConfig, csv_path: Optional[Path] = None, gpus: Optional[List[int]] = None, objectives: Optional[List[str]] = None):
    """Run final training on Pareto optimal configurations from NSGA-II results.
    
    Trains all Pareto optimal points using all available GPUs in parallel.
    
    Args:
        best_cfg: Best configuration found by NSGA-II (used as fallback if no CSV)
        csv_path: Path to nsga_evals.csv file. If None, uses NSGA_EVAL_CSV from config.
        gpus: List of GPU IDs to use. If None, uses FINAL_TRAIN_GPUS from config or all available.
        objectives: List of objective columns for Pareto front. If None, uses PARETO_OBJECTIVES from config.
    """
    print("\n" + "="*80)
    print("FINAL TRAINING - Pareto Optimal Configurations")
    print("="*80)
    
    # Determine CSV path
    log_utils.refresh_logging_paths()
    if csv_path is None:
        csv_path = Path(log_utils.NSGA_EVAL_CSV)
    else:
        csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"WARNING: NSGA evals CSV not found at {csv_path}")
        print("Falling back to single-config training...")
        _final_train_single(best_cfg)
        return
    
    # Determine run directory
    run_dir = csv_path.parent.resolve()
    cfg.set_dataset_log_dir(str(run_dir), create=False)
    os.environ["DATASET_LOG_DIR"] = str(run_dir)
    os.environ["NSGA_EVAL_CSV"] = str(run_dir / "nsga_evals.csv")
    os.environ["EPOCH_LOG_CSV"] = str(run_dir / "train_epoch_log.csv")
    os.environ["GEN_SUMMARY_CSV"] = str(run_dir / "nsga_gen_summary.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(run_dir / "checkpoint_validation.csv")
    os.environ["PROGRESS_LOG"] = str(run_dir / "progress.log")
    log_utils.refresh_logging_paths(str(run_dir))
    
    # Check GPU availability - use FINAL_TRAIN_GPUS from config if not specified
    if gpus is None:
        if FINAL_TRAIN_GPUS:
            # Use configured GPU list
            gpus = [int(g) for g in FINAL_TRAIN_GPUS]
        else:
            # Use all available GPUs
            gpus = None
    
    if gpus is None:
        num_visible = torch.cuda.device_count()
        available_gpus = list(range(num_visible)) if num_visible > 0 else []
    else:
        available_gpus = []
        for gpu_id in gpus:
            if gpu_id < torch.cuda.device_count():
                available_gpus.append(gpu_id)
            else:
                print(f"Warning: GPU {gpu_id} not available")
    
    if not available_gpus:
        print("No GPUs available, using CPU for final training...")
        available_gpus = [-1]  # CPU fallback
    
    print(f"Using GPUs: {available_gpus}")
    print(f"Loading evaluations from: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} evaluations")
    
    # Use objectives from config if not specified
    if objectives is None:
        objectives = PARETO_OBJECTIVES
    
    print(f"\nFinding Pareto optimal points using objectives: {objectives}")
    pareto_indices = _find_pareto_front(df, objectives)
    pareto_df = df.iloc[pareto_indices].sort_values('val_acc', ascending=False)
    
    print(f"\nFound {len(pareto_df)} Pareto optimal configurations:")
    print("-"*80)
    for idx, row in pareto_df.iterrows():
        print(f"  {row['eval_id']}:")
        print(f"    Accuracy: {row['val_acc']:.2f}%")
        print(f"    Circuit cost: {row['f2_circuit_cost']:.6f} s/sample")
        print(f"    Config: {row['embed']}, {int(row['n_qubits'])}q, depth={int(row['depth'])}")
        print()
    
    # Prepare configurations with descriptive names
    configs = []
    
    # Sort by accuracy to identify best/worst
    pareto_sorted_acc = pareto_df.sort_values('val_acc', ascending=False)
    pareto_sorted_cost = pareto_df.sort_values('f2_circuit_cost', ascending=True)
    
    best_acc_id = pareto_sorted_acc.iloc[0]['eval_id']
    lowest_cost_id = pareto_sorted_cost.iloc[0]['eval_id']
    
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
            'ent_ranges': _parse_ent_ranges(str(row['ent_ranges'])),
            'cnot_modes': _parse_cnot_modes(str(row['cnot_modes'])),
            'learning_rate': float(row['learning_rate']),
            'val_acc': row['val_acc'],
            'f2_circuit_cost': row['f2_circuit_cost'],
            'run_dir': str(run_dir)
        })
    
    print("-"*80)
    max_workers = len(available_gpus) * FINAL_WORKERS_PER_GPU
    print(f"\nStarting parallel training on {len(available_gpus)} GPUs "
          f"({FINAL_WORKERS_PER_GPU} worker(s)/GPU, {max_workers} total)...")
    run_folder_name = run_dir.name
    weights_dir = Path("weights") / run_folder_name
    print(f"Results will be saved to: {weights_dir}/")
    
    # Create final_training.csv file
    final_training_csv = run_dir / "final_training.csv"
    print(f"Final training results will be logged to: {final_training_csv}")
    print()
    
    # Initialize CSV file with header
    with open(final_training_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_TRAINING_HEADER)
        writer.writeheader()
    
    # Train in parallel using multiple GPUs
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, config in enumerate(configs):
            gpu_id = available_gpus[i % len(available_gpus)]
            future = executor.submit(_train_single_config, config, gpu_id)
            futures[future] = (config['eval_id'], gpu_id)
        
        # Collect results as they complete
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
                with open(final_training_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=FINAL_TRAINING_HEADER)
                    writer.writerow(row)
                    
            except Exception as e:
                print(f"\n✗ Error training {eval_id} on GPU {gpu_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL TRAINING SUMMARY")
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
    print("="*80)


def _final_train_single(best_cfg: QConfig):
    """Fallback: Run final training on a single best configuration.
    
    Args:
        best_cfg: Best configuration found by NSGA-II
    """
    from ..training.trainer import train_for_budget
    from ..utils.config import _update_worker_info
    
    print("\n== Final training on best config (single) ==")

    # Initialize status tracking for main process
    global STATUS_JSON_PATH, WORKER_GPU_ID, WORKER_RANK
    WORKER_GPU_ID = FINAL_TRAIN_GPU
    WORKER_RANK = -1  # Main process
    log_utils.refresh_logging_paths()
    STATUS_JSON_PATH = os.path.join(log_utils.STATUS_DIR, "main_final_training_status.json")

    # Update config module's worker variables
    _update_worker_info(WORKER_GPU_ID, WORKER_RANK)

    # Ensure PennyLane + Torch see only the chosen GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FINAL_TRAIN_GPU)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Final training device: {device_info} (single process, GPU={FINAL_TRAIN_GPU if device_info=='cuda' else '-'})")

    os.environ["QNAS_POOL_WORKER"] = "0"
    
    print(f"Final training data sizes: train={FINAL_TRAIN_SUBSET_SIZE if FINAL_TRAIN_SUBSET_SIZE > 0 else 'full'}, "
          f"val={FINAL_VAL_SUBSET_SIZE if FINAL_VAL_SUBSET_SIZE > 0 else 'full'}")
    
    # Create a new config with final training shots
    final_cfg = QConfig(
        embed_kind=best_cfg.embed_kind,
        n_qubits=best_cfg.n_qubits,
        depth=best_cfg.depth,
        ent_ranges=best_cfg.ent_ranges,
        cnot_modes=best_cfg.cnot_modes,
        learning_rate=best_cfg.learning_rate,
        shots=FINAL_SHOTS
    )
    
    vloss, vacc, model, _, _ = train_for_budget(final_cfg, "final-best", FINAL_TRAIN_EPOCHS, 0, 0, 
                                        train_size=FINAL_TRAIN_SUBSET_SIZE, val_size=FINAL_VAL_SUBSET_SIZE)
    # Save weights to weights/{run_folder}/
    run_folder_name = Path(cfg.DATASET_LOG_DIR).name if cfg.DATASET_LOG_DIR else "default"
    weights_dir = Path("weights") / run_folder_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"hybrid_qnn_best_{best_cfg.embed_kind}_nq{best_cfg.n_qubits}_d{best_cfg.depth}.pt"
    save_model_weights(model, str(save_path), final_cfg, eval_id="final-best", epoch=FINAL_TRAIN_EPOCHS, val_acc=vacc, val_loss=vloss)
    print(f"Final val: loss={vloss:.4f}, acc={vacc:.2f}% (backend={model.q_backend})")
    print(f"→ Saved weights: {save_path}")
    
    # Clean up status file
    try:
        if STATUS_JSON_PATH and os.path.exists(STATUS_JSON_PATH):
            os.remove(STATUS_JSON_PATH)
    except Exception:
        pass

"""NSGA-II optimization problem definition for Hybrid Quantum Neural Networks."""
import os
import time
from typing import Tuple

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from ..models.config import QConfig
from ..quantum.circuits import EMBED_ALL
from ..quantum.metrics import f3_num_subcircuits, _modes_str
from ..utils.logging_utils import _csv_append, _append_progress, _status_update, NSGA_EVAL_CSV, EVAL_HEADER, _get_worker_info
from ..utils.config import (
    NQ_MIN, NQ_MAX, DEPTH_MIN, DEPTH_MAX, ERANGE_MIN, ERANGE_MAX,
    CMODE_MIN, CMODE_MAX, LR_MIN, LR_MAX, SHOTS, ALLOWED_EMBEDDINGS,
    EVAL_EPOCHS, MAX_TRAIN_BATCHES, MAX_VAL_BATCHES, CUT_TARGET_QUBITS
)


def _get_gpu_id():
    """Get current worker GPU ID."""
    gpu_id, _ = _get_worker_info()
    return gpu_id


def _get_worker_rank():
    """Get current worker rank."""
    _, rank = _get_worker_info()
    return rank


# Counter for unique evaluation IDs
WORKER_EVAL_COUNTER = 0

# These are set by _mp_init in runner.py
CSV_LOCK = None
GLOBAL_COUNTER = None
CURRENT_GENERATION = None



def _next_eval_id() -> str:
    """Generate a unique evaluation ID."""
    global WORKER_EVAL_COUNTER
    WORKER_EVAL_COUNTER += 1
    g = None
    try:
        if GLOBAL_COUNTER is not None and CSV_LOCK is not None:
            with CSV_LOCK:
                GLOBAL_COUNTER.value += 1
                g = GLOBAL_COUNTER.value
    except Exception:
        pass
    base = f"{g:06d}" if g is not None else time.strftime("%H%M%S")
    return f"eval-{base}-g{_get_gpu_id()}-w{_get_worker_rank()}-{WORKER_EVAL_COUNTER:03d}"


# Global variables to cache fitted model parameters
_fitted_slope = None
_fitted_intercept = None
_fitted_model_loaded = False


def _fit_prediction_model_from_checkpoint_data(checkpoint_file: str, checkpoint_size: int = 16392) -> Tuple[float, float]:
    """
    Fit a linear regression model from checkpoint correlation data using Pearson correlation relationship.
    
    Loads checkpoint validation data, finds checkpoint size (default 16K), and fits:
    final_acc = slope * checkpoint_acc + intercept
    
    Args:
        checkpoint_file: Path to checkpoint_validation.csv
        checkpoint_size: Checkpoint size to use for fitting (default 16392 for 16K samples)
    
    Returns:
        Tuple of (slope, intercept) for the prediction model
    """
    try:
        import pandas as pd
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        
        df = pd.read_csv(checkpoint_file)
        
        # Get checkpoint epoch (typically epoch 1)
        checkpoint_epoch = 1
        if checkpoint_epoch not in df['epoch'].unique():
            checkpoint_epoch = sorted(df['epoch'].unique())[0]
        
        # Get final epoch (largest checkpoint size at same epoch, or epoch 10)
        final_epoch = checkpoint_epoch
        max_checkpoint_size = df['checkpoint_train_size'].max()
        
        # Get checkpoint data at specified size
        df_checkpoint = df[
            (df['epoch'] == checkpoint_epoch) & 
            (df['checkpoint_train_size'] == checkpoint_size)
        ][['eval_id', 'val_acc']].rename(columns={'val_acc': 'checkpoint_acc'})
        
        # Get final accuracy data (at max checkpoint size, same epoch)
        df_final = df[
            (df['epoch'] == final_epoch) & 
            (df['checkpoint_train_size'] == max_checkpoint_size)
        ][['eval_id', 'val_acc']].rename(columns={'val_acc': 'final_acc'})
        
        # Merge
        merged = pd.merge(df_checkpoint, df_final, on='eval_id', how='inner')
        
        if len(merged) < 2:
            raise ValueError(f"Insufficient data: only {len(merged)} samples for fitting")
        
        # Calculate Pearson correlation to verify relationship
        pearson_corr, pearson_p = pearsonr(merged['checkpoint_acc'], merged['final_acc'])
        
        # Fit linear regression model
        X = merged[['checkpoint_acc']].values
        y = merged['final_acc'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        
        _append_progress(f"[PREDICTION MODEL] Fitted from {checkpoint_file}: "
                        f"slope={slope:.4f}, intercept={intercept:.4f}, "
                        f"pearson_r={pearson_corr:.4f}, p_value={pearson_p:.6f}, n_samples={len(merged)}")
        
        return slope, intercept
        
    except Exception as e:
        _append_progress(f"[PREDICTION MODEL] Failed to fit from checkpoint data: {e}")
        raise


def _load_prediction_model() -> Tuple[float, float]:
    """
    Load or fit the prediction model parameters.
    
    Returns:
        Tuple of (slope, intercept) for the prediction model
    """
    global _fitted_slope, _fitted_intercept, _fitted_model_loaded
    
    if _fitted_model_loaded:
        return _fitted_slope, _fitted_intercept
    
    from ..utils.config import (
        PREDICTION_SLOPE, PREDICTION_INTERCEPT, 
        PREDICTION_MODEL_FILE, PREDICTION_MODEL_ENABLED,
        DATASET_LOG_DIR
    )
    
    # Try to load from file if specified
    if PREDICTION_MODEL_FILE:
        try:
            _fitted_slope, _fitted_intercept = _fit_prediction_model_from_checkpoint_data(PREDICTION_MODEL_FILE)
            _fitted_model_loaded = True
            return _fitted_slope, _fitted_intercept
        except Exception as e:
            _append_progress(f"[PREDICTION MODEL] Warning: Could not load from file {PREDICTION_MODEL_FILE}: {e}")
    
    # Try to auto-fit from current run's checkpoint data
    if PREDICTION_MODEL_ENABLED and DATASET_LOG_DIR:
        checkpoint_file = os.path.join(DATASET_LOG_DIR, "checkpoint_validation.csv")
        if os.path.exists(checkpoint_file):
            try:
                _fitted_slope, _fitted_intercept = _fit_prediction_model_from_checkpoint_data(checkpoint_file)
                _fitted_model_loaded = True
                return _fitted_slope, _fitted_intercept
            except Exception as e:
                _append_progress(f"[PREDICTION MODEL] Warning: Could not auto-fit from {checkpoint_file}: {e}")
    
    # Fall back to default parameters from config
    _fitted_slope = PREDICTION_SLOPE
    _fitted_intercept = PREDICTION_INTERCEPT
    _fitted_model_loaded = True
    _append_progress(f"[PREDICTION MODEL] Using default parameters: slope={_fitted_slope:.4f}, intercept={_fitted_intercept:.4f}")
    
    return _fitted_slope, _fitted_intercept


def predict_final_accuracy(checkpoint_acc: float) -> float:
    """
    Predict final accuracy from checkpoint accuracy using fitted linear model.
    
    Uses the relationship: final_acc = slope * checkpoint_acc + intercept
    
    Args:
        checkpoint_acc: Accuracy at checkpoint (e.g., epoch 1, 16K samples) in percentage (0-100)
    
    Returns:
        Predicted final accuracy in percentage (0-100)
    """
    slope, intercept = _load_prediction_model()
    predicted = slope * checkpoint_acc + intercept
    
    # Clamp to reasonable bounds (0-100%)
    predicted = max(0.0, min(100.0, predicted))
    
    return predicted


class QNNHyperProblem(ElementwiseProblem):
    """
    NSGA-II optimization problem for Hybrid Quantum Neural Networks.
    
    Decision vector:
      x[0]   = embed_id ∈ allowed {0..3}
      x[1]   = n_qubits ∈ [NQ_MIN, NQ_MAX]
      x[2]   = depth    ∈ [DEPTH_MIN, DEPTH_MAX]
      x[3:9]   = ent_ranges[0..5] ∈ [ERANGE_MIN, ERANGE_MAX] (use first 'depth')
      x[9:15]  = cnot_modes[0..5] ∈ {0,1,2,3} (use first 'depth')
      x[15]  = log10(lr) ∈ [log10(LR_MIN), log10(LR_MAX)]
      
    Objectives (minimize):
      f1 = 1 - accuracy (using actual accuracy at EVAL_EPOCHS)
      f2 = approximate circuit cost
      f3 = number of subcircuits after wire cutting (proxy for parallelizability)
         (only included when CUT_TARGET_QUBITS > 0)
    """
    
    def __init__(self, elementwise_runner=None):
        import math as _m
        ids = self.allowed_embed_ids()
        xl = [min(ids), NQ_MIN, DEPTH_MIN] + [ERANGE_MIN]*6 + [CMODE_MIN]*6 + [_m.log10(LR_MIN)]
        xu = [max(ids), NQ_MAX, DEPTH_MAX] + [ERANGE_MAX]*6 + [CMODE_MAX]*6 + [_m.log10(LR_MAX)]
        # Number of objectives: 2 if CUT_TARGET_QUBITS=0 (no wire cutting), 3 otherwise
        n_obj = 2 if CUT_TARGET_QUBITS <= 0 else 3
        super().__init__(n_var=16, n_obj=n_obj, n_constr=0, xl=np.array(xl), xu=np.array(xu), elementwise_runner=elementwise_runner)

    @staticmethod
    def allowed_embed_ids():
        """Get allowed embedding type IDs."""
        return [EMBED_ALL.index(e) for e in EMBED_ALL if e in ALLOWED_EMBEDDINGS]

    def _snap_embed(self, v):
        """Snap embedding value to nearest allowed ID."""
        idx = int(round(v))
        ids = self.allowed_embed_ids()
        if idx in ids:
            return idx
        return min(ids, key=lambda a: abs(a - idx))

    def _decode(self, x) -> QConfig:
        """Decode decision vector to QConfig."""
        embed_id = self._snap_embed(x[0])
        embed_kind = EMBED_ALL[embed_id]
        n_qubits = max(NQ_MIN, min(NQ_MAX, int(round(x[1]))))
        # Apply amplitude encoding qubit limit
        if embed_kind == "amplitude":
            n_qubits = min(n_qubits, 5)
        depth = max(DEPTH_MIN, min(DEPTH_MAX, int(round(x[2]))))
        ranges = []
        cap = max(1, n_qubits - 1)
        for i in range(6):
            r = max(ERANGE_MIN, min(ERANGE_MAX, int(round(x[3+i]))))
            r = min(r, cap)
            ranges.append(r)
        ent_ranges = ranges[:depth]
        modes = []
        for i in range(6):
            m = max(CMODE_MIN, min(CMODE_MAX, int(round(x[9+i]))))
            modes.append(m)
        cnot_modes = modes[:depth]
        lr = 10.0 ** float(x[15])
        return QConfig(embed_kind, n_qubits, depth, ent_ranges, cnot_modes, lr, SHOTS)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a single configuration."""
        # Import here to avoid circular dependency
        from ..training.trainer import train_for_budget
        
        eval_id = _next_eval_id()
        cfg = self._decode(x)

        # Get current generation from shared variable at the START of evaluation
        # Strategy: The callback sets counter to 'gen' (completed generation) after each generation.
        # When a new generation starts, the counter still has the previous generation's value.
        # We need to detect when we're in a new generation and increment accordingly.
        # Problem: All -001 evaluations in generation 0 run concurrently, so we can't use -001 as a signal.
        # Solution: Use a compare-and-swap approach - only increment if we successfully update from the old value.
        current_gen = -1
        if CURRENT_GENERATION is not None:
            try:
                with CSV_LOCK:
                    read_val = int(CURRENT_GENERATION.value)
                    
                    # If this is a -001 evaluation, we might be starting a new generation
                    # Use compare-and-swap: try to increment from read_val to read_val+1
                    # Only one evaluation will succeed (the first one to read the value)
                    if eval_id.endswith('-001') and read_val >= 0:
                        # Try to increment: set to read_val + 1
                        # This is safe because we're in a lock, so only one thread can do this at a time
                        # But wait - multiple processes! The lock is shared, so this should still work.
                        # Actually, the issue is that multiple -001 evaluations in gen 0 all read 0,
                        # so they all try to set it to 1. We need to ensure only one succeeds.
                        # Since we're in a lock, only one process can execute this at a time.
                        # So if we read 0 and set to 1, the next one will read 1 and set to 2.
                        # That's still wrong!
                        
                        # Better: Check if we're actually in a new generation by comparing
                        # the counter value to what we expect. But we don't know what we expect!
                        
                        # Actually, the simplest fix: Don't increment for generation 0.
                        # Generation 0 evaluations should read 0 (which is correct).
                        # Only increment when we detect we're in generation 1 or later.
                        # How? If read_val is 0 and this is the first time we're seeing it, stay at 0.
                        # If read_val is N > 0, we're in generation N+1, so increment to N+1.
                        
                        if read_val == 0:
                            # Generation 0: stay at 0
                            current_gen = 0
                        else:
                            # Generation read_val+1: increment to read_val+1
                            current_gen = read_val + 1
                            CURRENT_GENERATION.value = current_gen
                            _append_progress(f"[DEBUG {eval_id}] New generation detected: counter {read_val} -> {current_gen}")
                    else:
                        # Not a -001 evaluation, use the current counter value
                        current_gen = read_val
            except Exception as e:
                # If reading fails, default to -1 (unknown generation)
                _append_progress(f"[DEBUG {eval_id}] Failed to read generation: {e}")
                pass

        _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] queued evaluation")
        _status_update({"stage": "eval_start", "eval_id": eval_id})
        t0 = time.time()
        try:
            vloss, vacc, _model, val_time, val_samples = train_for_budget(cfg, eval_id, EVAL_EPOCHS,
                                                MAX_TRAIN_BATCHES, MAX_VAL_BATCHES)
            secs = time.time() - t0
            
            # Use actual accuracy at EVAL_EPOCHS for f1 (1 - accuracy)
            f1 = 1.0 - (vacc / 100.0)
            
            # Calculate time-based circuit cost: time per sample on validation dataset
            # If val_samples=5000 and val_time=10000s, then cost = 10000/5000 = 2 seconds/sample
            if val_samples <= 0:
                raise RuntimeError(
                    f"F2 calculation failed: No validation samples were processed (val_samples={val_samples}). "
                    f"This is required for time-based cost calculation. "
                    f"Check that validation dataset is properly loaded and processed."
                )
            f2 = val_time / val_samples  # Time per sample (seconds)
            # Calculate f3 (number of subcircuits after wire cutting)
            # f3 is always calculated and logged to CSV, but only included in objectives when CUT_TARGET_QUBITS > 0
            f3 = f3_num_subcircuits(cfg.n_qubits, cfg.depth, cfg.ent_ranges, cfg.cnot_modes, CUT_TARGET_QUBITS)
            
            # Note: current_gen was already read at the start of evaluation (above)
            # This ensures we capture the generation when the evaluation began, not when it finished
            
            _csv_append(NSGA_EVAL_CSV, {
                "eval_id": eval_id, "gen_est": current_gen,
                "embed": cfg.embed_kind, "n_qubits": cfg.n_qubits, "depth": cfg.depth,
                "ent_ranges": "-".join(map(str, cfg.ent_ranges)),
                "cnot_modes": _modes_str(cfg.cnot_modes),
                "learning_rate": f"{cfg.learning_rate:.6e}",
                "val_loss": f"{vloss:.6f}", "val_acc": f"{vacc:.4f}",
                "f1_1_minus_acc": f"{f1:.6f}", "f2_circuit_cost": f"{f2:.6f}",  # Time-based cost (seconds per sample)
                "f3_n_subcircuits": f"{int(f3)}",
                "q_backend": _model.q_backend, "seconds": f"{secs:.2f}",
                "gpu_id": _get_gpu_id(), "worker_rank": _get_worker_rank(), "pid": os.getpid()
            }, EVAL_HEADER)
            _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] DONE in {secs:.1f}s | val_acc={vacc:.2f}% | time_cost={f2:.6f}s/sample | n_sub={f3}")
            _status_update({"stage": "eval_done", "eval_id": eval_id, "val_acc": vacc, "val_loss": vloss, "seconds": secs})
            
            # Include f3 in objectives only when CUT_TARGET_QUBITS > 0
            if CUT_TARGET_QUBITS > 0:
                out["F"] = np.array([f1, float(f2), float(f3)], dtype=float)
            else:
                out["F"] = np.array([f1, float(f2)], dtype=float)
        except Exception as ex:
            secs = time.time() - t0
            _append_progress(f"[GPU{_get_gpu_id()}|{eval_id}] ERROR after {secs:.1f}s: {repr(ex)}")
            _status_update({"stage": "eval_error", "eval_id": eval_id, "error": str(ex), "seconds": secs})
            # Error penalty values match the number of objectives
            if CUT_TARGET_QUBITS > 0:
                out["F"] = np.array([1.0, 1e9, 1e6], dtype=float)
            else:
                out["F"] = np.array([1.0, 1e9], dtype=float)


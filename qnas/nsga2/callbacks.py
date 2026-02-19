"""NSGA-II progress callbacks for logging and monitoring."""
import time

import numpy as np
from pymoo.core.callback import Callback

from ..utils import logging_utils as log_utils
from .problem import CSV_LOCK, CURRENT_GENERATION


class ProgressCallback(Callback):
    """Callback for logging NSGA-II progress after each generation."""
    
    def __init__(self, generation_proxy=None, csv_lock=None):
        super().__init__()
        self.t0 = time.time()
        self.generation_proxy = generation_proxy
        self.csv_lock = csv_lock
    
    def notify(self, algorithm):
        """Called after each generation completes."""
        gen = algorithm.n_gen

        # Update generation counter first; evaluations will read this snapshot.
        gen_proxy = self.generation_proxy if self.generation_proxy is not None else CURRENT_GENERATION
        lock = self.csv_lock if self.csv_lock is not None else CSV_LOCK

        try:
            if gen_proxy is not None:
                if lock is not None:
                    with lock:
                        gen_proxy.value = int(gen)
                else:
                    gen_proxy.value = int(gen)
            else:
                log_utils._append_progress(f"[CALLBACK] Gen {gen}: no generation proxy available")
        except Exception as exc:
            log_utils._append_progress(f"[CALLBACK] Gen {gen}: failed to update generation counter: {exc}")
        
        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return
        
        # Only print if we have at least one valid evaluation (not all penalty values)
        # Penalty values are: f1=1.0, f2=1e9, f3=1e6 (if included)
        valid_evals = F[(F[:,0] < 1.0) | (F[:,1] < 1e9)]
        if len(valid_evals) == 0:
            # All evaluations failed, skip printing
            return
        
        best = int(np.argmin(F[:,0]))
        elapsed = (time.time() - self.t0) / 60.0
        # Handle variable number of objectives (f3 only present when CUT_TARGET_QUBITS > 0)
        best_n_sub = int(F[best,2]) if F.shape[1] > 2 else -1
        n_sub_str = f" | n_sub={best_n_sub}" if F.shape[1] > 2 else ""
        msg = f"[Gen {gen:02d}] best (1-acc)={F[best,0]:.4f} | time_cost={F[best,1]:.6f}s/sample{n_sub_str} | median (1-acc)={np.median(F[:,0]):.4f} | elapsed={elapsed:.1f}m"
        print(msg)
        log_utils._append_progress(msg)
        
        # Write per-generation summary row
        log_utils._csv_append(log_utils.GEN_SUMMARY_CSV, {
            "generation": int(gen),
            "best_1_minus_acc": f"{F[best,0]:.6f}",
            "best_cost": f"{F[best,1]:.6f}",  # Time-based cost (seconds per sample)
            "best_n_subcircuits": f"{best_n_sub}",  # -1 if f3 not included in objectives
            "median_1_minus_acc": f"{np.median(F[:,0]):.6f}",
            "elapsed_minutes": f"{elapsed:.2f}",
        }, log_utils.GEN_HEADER)

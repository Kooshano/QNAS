"""NSGA-II progress callbacks for logging and monitoring."""
import time

import numpy as np
from pymoo.core.callback import Callback

from ..utils.logging_utils import _csv_append, _append_progress, GEN_SUMMARY_CSV, GEN_HEADER
from .problem import CSV_LOCK, CURRENT_GENERATION


class ProgressCallback(Callback):
    """Callback for logging NSGA-II progress after each generation."""
    
    def __init__(self, generation_proxy=None, csv_lock=None):
        super().__init__()
        self.t0 = time.time()
        # Store references to shared memory objects
        self.generation_proxy = generation_proxy
        self.csv_lock = csv_lock
        # Track if we've seen generation 0 (pymoo may not call callback for gen 0)
        self.seen_gen_0 = False
    
    def notify(self, algorithm):
        """Called after each generation completes."""
        gen = algorithm.n_gen
        
        # CRITICAL: Update generation counter FIRST, before any other operations
        # This callback is called AFTER generation 'gen' completes.
        # IMPORTANT: pymoo may not call this callback for generation 0 (initial population).
        # If this is the first callback we see and gen > 0, we need to handle generation 0.
        # Use stored proxy if available, otherwise fall back to global
        gen_proxy = self.generation_proxy if self.generation_proxy is not None else CURRENT_GENERATION
        lock = self.csv_lock if self.csv_lock is not None else CSV_LOCK
        
        # Handle case where generation 0 callback didn't run
        if not self.seen_gen_0 and gen > 0:
            # This is the first callback, and it's for generation > 0
            # This means generation 0's callback didn't run
            # We should have set counter to 1 after gen 0, but since callback didn't run,
            # the counter is still at 0. So when gen 1 completes, we see counter=0.
            # We'll set it to gen+1 (which is 2 for gen 1), but this means gen 1 evaluations
            # that are still writing will read 2 instead of 1.
            # The best we can do is set it to gen (1) first, then gen+1 (2)
            if gen_proxy is not None and lock is not None:
                try:
                    with lock:
                        gen_proxy.value = int(gen)
                except Exception:
                    pass
            self.seen_gen_0 = True
        
        if gen_proxy is None:
            error_msg = f"[CALLBACK] Gen {gen}: generation_proxy is None! Cannot update counter."
            print(error_msg)
            _append_progress(error_msg)
        elif lock is None:
            error_msg = f"[CALLBACK] Gen {gen}: csv_lock is None! Cannot update counter safely."
            print(error_msg)
            _append_progress(error_msg)
        else:
            try:
                # Update with lock to ensure thread safety
                with lock:
                    # Set counter to the generation that just completed (gen)
                    gen_proxy.value = int(gen)
                    # Handle missing generation 0 callback
                    if not self.seen_gen_0:
                        self.seen_gen_0 = True
            except Exception as e:
                # Log error but don't fail
                import sys
                import traceback
                error_msg = f"[CALLBACK] ERROR updating counter at gen {gen}: {e}"
                traceback_msg = traceback.format_exc()
                print(error_msg, file=sys.stderr)
                print(traceback_msg, file=sys.stderr)
                _append_progress(error_msg)
                _append_progress(traceback_msg)
        
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
        _append_progress(msg)
        
        # Write per-generation summary row
        try:
            _csv_append(GEN_SUMMARY_CSV, {
                "generation": int(gen),
                "best_1_minus_acc": f"{F[best,0]:.6f}",
                "best_cost": f"{F[best,1]:.6f}",  # Time-based cost (seconds per sample)
                "best_n_subcircuits": f"{best_n_sub}",  # -1 if f3 not included in objectives
                "median_1_minus_acc": f"{np.median(F[:,0]):.6f}",
                "elapsed_minutes": f"{elapsed:.2f}",
            }, GEN_HEADER)
        except Exception:
            pass


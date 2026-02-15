#!/usr/bin/env python3
"""Correlation-aware NSGA-II retraining utility (Main-compatible logging, 2 workers/GPU).

- Logs exactly like your Main module (qnas.main): same epoch CSV schema and a shared progress.log.
- Default scheduling: TWO workers per GPU, each handling one QNAS retrain job.
- Uses ALL visible GPUs unless --max-gpus limits it.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import numpy as np               # noqa
import pandas as pd              # noqa
import multiprocessing as mp     # noqa


# --------------------------
# Args & small helpers
# --------------------------

def _nonneg_int_or_none(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return v

# Add project root to path (go up 3 levels from scripts/analysis/)
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

DEFAULT_HQ_MODULE_CANDIDATES = ("qnas.main", "qnas_nsga2_single", "Main")


def _load_env_bool(key: str, default: bool) -> bool:
    """Load boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _load_env_int(key: str, default: int) -> int:
    """Load integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _load_env_float(key: str, default: Optional[float]) -> Optional[float]:
    """Load float from environment variable."""
    val = os.getenv(key, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain selected NSGA-II candidates and study correlation (Main-compatible logging).")
    p.add_argument("log_dir", type=Path, nargs='?', default=None, 
                   help="Path to dated NSGA-II log folder, e.g. logs/mnist_2025-10-05. Omit to generate random samples.")
    p.add_argument("--analyze-run", type=str, metavar="RUN_ID", 
                   help="Analyze a specific run directory (e.g., run_20251006-163458) to extract epoch-by-epoch correlation data")
    p.add_argument("--analyze-epochs", type=str, metavar="EPOCHS",
                   help="Comma-separated list of epochs to analyze (e.g., '1,5,10,20'). If not specified, analyzes all epochs.")
    p.add_argument("--show-final-accuracy", action="store_true",
                   help="Show final training accuracy for each candidate at specified epochs")
    p.add_argument("--random-mode", action="store_true",
                   help="Generate random architecture samples and train them (no NSGA-II logs required)")
    p.add_argument("--random-samples", type=int, default=10, metavar="N",
                   help="Number of random samples to generate in random mode (default: 10)")
    return p.parse_args(argv)


def _resolve_hq_module_name(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for candidate in DEFAULT_HQ_MODULE_CANDIDATES:
        try:
            if importlib.util.find_spec(candidate) is not None:
                return candidate
        except Exception:
            continue
    raise ImportError(
        "Could not locate a QNAS training module. Set --hq-module (e.g. 'Main') or ensure "
        "qnas.main is on PYTHONPATH."
    )

def infer_dataset_name(log_dir: Path) -> Optional[str]:
    """Infer dataset from path. Supports:
    - logs/nsga-ii/DATASET/run_TIMESTAMP -> use parent folder (e.g. HEART-FAILURE -> heart-failure)
    - logs/dataset_date (e.g. mnist_2025-10-05) -> use folder name before first underscore (mnist)
    """
    name = log_dir.name.lower()
    if name.startswith("run_"):
        return log_dir.parent.name.lower().replace("_", "-")
    return name.split("_", 1)[0] if "_" in name else None

def _copy_env_to_run_folder(run_dir: Path, stamp: str) -> None:
    """Copy .env to run folder as config_{stamp}.env for reproducibility (no .env in run dir)."""
    import shutil
    env_file = Path(".env")
    if env_file.exists():
        try:
            dest_env_named = run_dir / f"config_{stamp}.env"
            shutil.copy2(env_file, dest_env_named)
        except Exception as e:
            print(f"[WARN] Could not copy .env to run folder: {e}", file=sys.stderr)


def ensure_required_files(log_dir: Path) -> None:
    """Check for required files. Only train_epoch_log.csv is required for correlation analysis.
    nsga_evals.csv is optional - only needed if selecting candidates from NSGA-II results."""
    required = [log_dir / "train_epoch_log.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file: " + ", ".join(missing))
    
    # Warn if nsga_evals.csv is missing (but don't fail)
    if not (log_dir / "nsga_evals.csv").exists():
        print(f"[WARNING] nsga_evals.csv not found in {log_dir}")
        print(f"[WARNING] This file is only needed if selecting candidates from NSGA-II results")
        print(f"[WARNING] For correlation analysis of existing training data, it's not required")


# --------------------------
# Data containers
# --------------------------

@dataclass
class Candidate:
    eval_id: str
    generation: int
    embed: str
    n_qubits: int
    depth: int
    ent_ranges: List[int]
    cnot_modes: List[int]
    learning_rate: float
    shots: int
    selection: str
    pre_val_acc: float
    pre_val_loss: float
    original_learning_rate: float
    meta: Dict[str, float]

@dataclass
class RetrainResult:
    eval_id: str
    post_val_acc: Optional[float]
    post_val_loss: Optional[float]
    epochs: int
    lr_used: float
    shots_used: int
    device: str
    seconds: Optional[float]
    weights_path: Optional[str]
    error: Optional[str] = None


# --------------------------
# Candidate selection
# --------------------------

def _ensure_parameter_diversity(candidates: List[Candidate], min_diversity_ratio: float = 0.7) -> List[Candidate]:
    """Ensure selected candidates have sufficient parameter diversity.
    
    Args:
        candidates: List of candidate configurations
        min_diversity_ratio: Minimum ratio of unique parameter combinations required
    
    Returns:
        Filtered list with improved diversity
    """
    if len(candidates) <= 1:
        return candidates
    
    # Create parameter fingerprints for each candidate
    param_fingerprints = []
    for c in candidates:
        fingerprint = (
            c.embed,
            c.n_qubits, 
            c.depth,
            tuple(c.ent_ranges) if c.ent_ranges else (),
            tuple(c.cnot_modes) if c.cnot_modes else (),
            round(c.learning_rate, 6),  # Round to avoid floating point issues
            c.shots
        )
        param_fingerprints.append(fingerprint)
    
    # Count unique configurations
    unique_fingerprints = set(param_fingerprints)
    diversity_ratio = len(unique_fingerprints) / len(candidates)
    
    print(f"[DIVERSITY] Parameter diversity: {len(unique_fingerprints)}/{len(candidates)} "
          f"unique configs ({diversity_ratio:.2%})")
    
    if diversity_ratio < min_diversity_ratio:
        print(f"[DIVERSITY] Warning: Low parameter diversity ({diversity_ratio:.2%} < {min_diversity_ratio:.2%})")
        print(f"[DIVERSITY] Consider adjusting selection parameters for more diverse samples")
    
    # Remove exact duplicates while preserving order
    # But always keep special selections like "best_overall" and "final_best"
    seen_fingerprints = set()
    diverse_candidates = []
    protected_selections = {"best_overall", "final_best"}
    
    for i, candidate in enumerate(candidates):
        fingerprint = param_fingerprints[i]
        # Never remove protected selections (best_overall, final_best), even if they have duplicate configs
        if fingerprint not in seen_fingerprints or candidate.selection in protected_selections:
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
            diverse_candidates.append(candidate)
        else:
            print(f"[DIVERSITY] Removing duplicate config: {candidate.eval_id}")
    
    return diverse_candidates

def _analyze_sample_diversity(candidates: List[Candidate]) -> Dict[str, float]:
    """Analyze diversity across different parameter dimensions."""
    if not candidates:
        return {}
    
    n = len(candidates)
    diversity_metrics = {
        'embed_diversity': len(set(c.embed for c in candidates)) / n,
        'qubit_diversity': len(set(c.n_qubits for c in candidates)) / n,
        'depth_diversity': len(set(c.depth for c in candidates)) / n,
        'lr_diversity': len(set(round(c.learning_rate, 6) for c in candidates)) / n,
        'shots_diversity': len(set(c.shots for c in candidates)) / n,
        'generation_diversity': len(set(c.generation for c in candidates)) / n,
    }
    
    print(f"[DIVERSITY] Sample diversity analysis:")
    for metric, value in diversity_metrics.items():
        print(f"[DIVERSITY]   {metric}: {value:.2%}")
    
    return diversity_metrics

def _is_valid_architecture(embed: str, n_qubits: int, depth: int, ent_ranges: List[int], 
                           dataset: str = None) -> Tuple[bool, str]:
    """Validate architecture configuration to prevent known failure modes.
    
    Args:
        embed: Embedding type (angle-x, angle-y, angle-z, amplitude)
        n_qubits: Number of qubits
        depth: Circuit depth
        ent_ranges: Entanglement ranges for each layer
        dataset: Dataset name for dataset-specific rules
    
    Returns:
        (is_valid, reason) tuple
    """
    embed = embed.lower()
    
    # Detect complex datasets (color images with high dimensionality)
    is_complex_dataset = dataset and dataset.lower() in ['cifar10', 'cifar100', 'svhn', 'imagenet']
    
    # Rule 1: angle-z requires reasonable depth and entanglement
    if embed == 'angle-z':
        # RZ + RY gates can work with moderate depth
        if depth < 2:
            return False, f"angle-z with depth={depth} < 2: needs at least depth=2 for feature learning"
        
        # Check if entanglement is reasonable
        avg_ent_range = sum(ent_ranges) / len(ent_ranges) if ent_ranges else 0
        if avg_ent_range < 2:
            return False, f"angle-z with avg_ent_range={avg_ent_range:.1f} < 2: needs some entanglement for correlation"
        
        # Less strict for complex datasets now with improved measurement
        if is_complex_dataset:
            if depth < 3:
                return False, f"angle-z on {dataset} needs depth≥3, got depth={depth}"
            if avg_ent_range < 2.5:
                return False, f"angle-z on {dataset} needs avg_ent_range≥2.5, got {avg_ent_range:.1f}"
    
    # Rule 2: amplitude encoding requires sufficient qubits and depth for complex data
    if embed == 'amplitude':
        target_amplitudes = 2 ** n_qubits
        
        if is_complex_dataset:
            # CIFAR10: 3072 features, need reasonable compression ratio
            if n_qubits < 5:
                compression_ratio = 3072 / target_amplitudes
                return False, f"amplitude on {dataset} with {n_qubits} qubits: compression={compression_ratio:.0f}x too extreme (need ≥5 qubits)"
            
            if depth < 3:
                return False, f"amplitude on {dataset} with depth={depth} < 3: shallow circuits can't recover from compression"
        else:
            # MNIST: 784 features, more lenient
            if n_qubits < 5:
                return False, f"amplitude with {n_qubits} qubits gives only {target_amplitudes} amplitudes (need ≥5 qubits)"
            
            if depth < 2:
                return False, f"amplitude with depth={depth} < 2: need deeper circuits to process compressed features"
    
    # Rule 3: Very shallow circuits generally fail
    if depth == 1 and embed != 'angle-y':
        # angle-y is most stable for shallow circuits due to real amplitudes
        if n_qubits > 6:
            return False, f"depth=1 with {n_qubits} qubits: too shallow for complex feature space"
    
    return True, "valid"


def generate_random_candidates(
    n_samples: int,
    lr_override: Optional[float],
    shots: int,
    seed: int,
    HQMain,
) -> List[Candidate]:
    """Generate random architecture candidates for training.
    
    Args:
        n_samples: Number of random candidates to generate
        lr_override: Optional fixed learning rate (None = random)
        shots: Number of shots for quantum differentiation
        seed: Random seed for reproducibility
        HQMain: Imported training module with configuration constants
    
    Returns:
        List of randomly generated Candidate objects
    """
    print(f"[RANDOM_MODE] Generating {n_samples} random architecture samples")
    print(f"[RANDOM_MODE] Using seed {seed} for reproducibility")
    
    rng = np.random.default_rng(seed)
    
    # Get configuration ranges from Main module
    allowed_embeddings = getattr(HQMain, "ALLOWED_EMBEDDINGS", ["angle-x", "angle-y", "angle-z", "amplitude"])
    if isinstance(allowed_embeddings, str):
        allowed_embeddings = [e.strip() for e in allowed_embeddings.split(",")]
    
    # Get dataset for validation rules
    dataset = getattr(HQMain, "DATASET", "unknown")
    
    nq_min = int(getattr(HQMain, "NQ_MIN", 2))
    nq_max = int(getattr(HQMain, "NQ_MAX", 12))
    depth_min = int(getattr(HQMain, "DEPTH_MIN", 1))
    depth_max = int(getattr(HQMain, "DEPTH_MAX", 6))
    erange_min = int(getattr(HQMain, "ERANGE_MIN", 1))
    erange_max = int(getattr(HQMain, "ERANGE_MAX", 6))
    lr_min = float(getattr(HQMain, "LR_MIN", 0.001))
    lr_max = float(getattr(HQMain, "LR_MAX", 0.005))
    cnot_modes = getattr(HQMain, "CNOT_MODES", ["all", "odd", "even", "none"])
    
    print(f"[RANDOM_MODE] Parameter ranges:")
    print(f"[RANDOM_MODE]   Dataset: {dataset}")
    print(f"[RANDOM_MODE]   Embeddings: {allowed_embeddings}")
    print(f"[RANDOM_MODE]   Qubits: {nq_min}-{nq_max} (amplitude limited to max 5)")
    print(f"[RANDOM_MODE]   Depth: {depth_min}-{depth_max}")
    print(f"[RANDOM_MODE]   Entanglement range: {erange_min}-{erange_max}")
    print(f"[RANDOM_MODE]   Learning rate: {lr_min}-{lr_max}")
    print(f"[RANDOM_MODE]   Validation: Enabled (filtering problematic configs)")
    
    candidates = []
    rejected_count = 0
    max_attempts = n_samples * 100  # Prevent infinite loop
    attempts = 0
    
    while len(candidates) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Randomly sample architecture parameters
        embed = rng.choice(allowed_embeddings)
        n_qubits = int(rng.integers(nq_min, nq_max + 1))
        
        # Apply amplitude embedding qubit limit (max 5 qubits)
        if embed.lower() == "amplitude":
            n_qubits = min(n_qubits, 5)
        
        depth = int(rng.integers(depth_min, depth_max + 1))
        
        # Generate random entanglement ranges for each layer
        ent_ranges = [int(rng.integers(erange_min, min(erange_max, n_qubits) + 1)) for _ in range(depth)]
        
        # Validate architecture before accepting
        is_valid, reason = _is_valid_architecture(embed, n_qubits, depth, ent_ranges, dataset)
        
        if not is_valid:
            rejected_count += 1
            if rejected_count <= 5:  # Only show first few rejections to avoid spam
                print(f"[RANDOM_MODE] Rejected config: {reason}")
            continue
        
        # Generate random CNOT modes for each layer
        cnot_mode_indices = [int(rng.integers(0, len(cnot_modes))) for _ in range(depth)]
        
        # Sample learning rate
        if lr_override is not None:
            learning_rate = float(lr_override)
        else:
            learning_rate = float(rng.uniform(lr_min, lr_max))
        
        # Create candidate
        eval_id = f"random-{len(candidates):04d}"
        candidates.append(Candidate(
            eval_id=eval_id,
            generation=-1,  # Not from NSGA-II
            embed=embed.lower(),
            n_qubits=n_qubits,
            depth=depth,
            ent_ranges=ent_ranges,
            cnot_modes=cnot_mode_indices,
            learning_rate=learning_rate,
            shots=shots,
            selection="random_sample",
            pre_val_acc=0.0,  # No pre-training
            pre_val_loss=float('inf'),  # No pre-training
            original_learning_rate=learning_rate,
            meta={
                "f2_circuit_cost": float('nan'),  # Unknown before training
                "f3_n_subcircuits": float('nan'),  # Unknown before training
                "seconds": float('nan'),
            },
        ))
    
    if attempts >= max_attempts:
        print(f"[RANDOM_MODE] WARNING: Hit max attempts ({max_attempts}), only generated {len(candidates)}/{n_samples} candidates")
        print(f"[RANDOM_MODE] Consider relaxing validation constraints or expanding parameter ranges")
    
    print(f"[RANDOM_MODE] Generated {len(candidates)} valid candidates (rejected {rejected_count} invalid configs)")
    
    if rejected_count > 5:
        print(f"[RANDOM_MODE] ... and {rejected_count - 5} more rejections (not shown)")
    
    # Analyze diversity
    _analyze_sample_diversity(candidates)
    
    return candidates

def load_nsga_logs(log_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load NSGA-II logs. nsga_evals.csv is optional - only needed for candidate selection."""
    nsga_evals_path = log_dir / "nsga_evals.csv"
    if nsga_evals_path.exists():
        evals = pd.read_csv(nsga_evals_path)
        evals["generation"] = evals.get("gen_est", -1).fillna(-1).astype(int)
        evals["val_acc"] = pd.to_numeric(evals["val_acc"], errors="coerce")
        evals["val_loss"] = pd.to_numeric(evals["val_loss"], errors="coerce")
        evals["learning_rate"] = pd.to_numeric(evals["learning_rate"], errors="coerce")
        evals = evals.dropna(subset=["val_acc", "learning_rate"])
    else:
        # Return empty DataFrame if nsga_evals.csv doesn't exist
        print(f"[INFO] nsga_evals.csv not found - returning empty DataFrame")
        evals = pd.DataFrame()
    
    epochs = pd.read_csv(log_dir / "train_epoch_log.csv")
    return evals, epochs

def _parse_int_list(raw: str, expected: int, fallback: int) -> List[int]:
    if pd.isna(raw) or raw == "": return [fallback] * expected
    parts = [p.strip() for p in str(raw).split("-") if p.strip()]
    vals: List[int] = []
    for p in parts[:expected]:
        try: vals.append(int(p))
        except ValueError: vals.append(fallback)
    if not vals: vals = [fallback] * expected
    while len(vals) < expected: vals.append(vals[-1])
    return vals

def _parse_cnot_modes(raw: str, expected: int, lookup: Dict[str, int]) -> List[int]:
    if pd.isna(raw) or raw == "": return [lookup.get("none", 3)] * expected
    parts = [p.strip().lower() for p in str(raw).split("-") if p.strip()]
    vals = [lookup.get(p, lookup.get("none", 3)) for p in parts[:expected]]
    if not vals: vals = [lookup.get("none", 3)] * expected
    while len(vals) < expected: vals.append(vals[-1])
    return vals

def select_candidates(
    evals: pd.DataFrame,
    random_per_gen: int,
    extra_random: int,
    lr_override: Optional[float],
    lr_scale: float,
    shots: int,
    cnot_lookup: Dict[str, int],
    seed: int,
    epoch_df: Optional[pd.DataFrame] = None,
) -> List[Candidate]:
    print(f"[SELECTION] Using seed {seed} for reproducible sampling")
    rng = np.random.default_rng(seed)
    selected_rows: List[pd.Series] = []
    reasons: Dict[str, str] = {}

    # Always select best from each generation
    best_per_gen = (
        evals.sort_values(["generation", "val_acc"], ascending=[True, False])
             .groupby("generation", as_index=False).first()
    )
    print(f"[SELECTION] Selected {len(best_per_gen)} best candidates (1 per generation)")
    for _, row in best_per_gen.iterrows():
        selected_rows.append(row); reasons[row["eval_id"]] = "best_generation"

    # Add random candidates from each generation (ensuring diversity)
    if random_per_gen:
        total_random_gen = 0
        for gen, group in evals.groupby("generation"):
            # Exclude already selected candidates
            available = group[~group["eval_id"].isin(reasons)]
            if available.empty:
                continue
            
            # Shuffle with different seed per generation for better diversity
            gen_seed = rng.integers(0, 1 << 32)
            available = available.sample(frac=1.0, random_state=gen_seed)
            
            n_to_select = min(random_per_gen, len(available))
            for _, row in available.head(n_to_select).iterrows():
                selected_rows.append(row); reasons[row["eval_id"]] = "random_generation"
                total_random_gen += 1
        print(f"[SELECTION] Selected {total_random_gen} random candidates from generations")

    # Add extra random candidates globally
    if extra_random:
        remaining = evals[~evals["eval_id"].isin(reasons)]
        if not remaining.empty:
            n_to_select = min(extra_random, len(remaining))
            sample = remaining.sample(
                n=n_to_select,
                random_state=rng.integers(0, 1 << 32),
            )
            for _, row in sample.iterrows():
                selected_rows.append(row); reasons[row["eval_id"]] = "random_global"
            print(f"[SELECTION] Selected {len(sample)} additional random candidates globally")

    # Remove duplicates (shouldn't happen but safety check)
    seen = set()
    unique_rows: List[pd.Series] = []
    duplicates_removed = 0
    for row in selected_rows:
        eid = row["eval_id"]
        if eid in seen: 
            duplicates_removed += 1
            continue
        seen.add(eid); unique_rows.append(row)
    
    if duplicates_removed > 0:
        print(f"[SELECTION] Removed {duplicates_removed} duplicate eval_ids")

    # Convert to Candidate objects
    cands: List[Candidate] = []
    for row in unique_rows:
        depth = int(row["depth"])
        ent_ranges = _parse_int_list(row.get("ent_ranges", ""), depth, 1)
        cnot_modes = _parse_cnot_modes(row.get("cnot_modes", ""), depth, cnot_lookup)
        base_lr = float(row["learning_rate"])
        final_lr = float(lr_override) if lr_override is not None else base_lr * lr_scale
        cands.append(Candidate(
            eval_id=str(row["eval_id"]),
            generation=int(row["generation"]),
            embed=str(row["embed"]).lower(),
            n_qubits=int(row["n_qubits"]),
            depth=depth,
            ent_ranges=ent_ranges,
            cnot_modes=cnot_modes,
            learning_rate=final_lr,
            shots=int(shots),
            selection=reasons[row["eval_id"]],
            pre_val_acc=float(row["val_acc"]),
            pre_val_loss=float(row.get("val_loss", np.nan)),
            original_learning_rate=base_lr,
            meta={
                "f2_circuit_cost": float(row.get("f2_circuit_cost", np.nan)),
                "f3_n_subcircuits": float(row.get("f3_n_subcircuits", np.nan)),
                "seconds": float(row.get("seconds", np.nan)),
            },
        ))

    # Always include the overall best model from NSGA-II (the "best of the best")
    if not evals.empty:
        # Find the model with the highest validation accuracy
        best_overall = evals.loc[evals["val_acc"].idxmax()]
        best_eval_id = best_overall["eval_id"]
        
        # Check if it's already selected
        if best_eval_id in reasons:
            # If already selected, just note that it's the best overall
            for candidate in cands:
                if candidate.eval_id == best_eval_id:
                    print(f"[SELECTION] Overall best model {best_eval_id} already selected as: {reasons[best_eval_id]}")
                    break
        else:
            # Add it as a new candidate
            selected_rows.append(best_overall)
            reasons[best_eval_id] = "best_overall"
            print(f"[SELECTION] Selected overall best model: {best_eval_id} (acc={best_overall['val_acc']:.2f}%)")
            # Need to recreate cands if we added a new row
            # Convert to Candidate objects again
            seen = set()
            unique_rows: List[pd.Series] = []
            for row in selected_rows:
                eid = row["eval_id"]
                if eid not in seen:
                    seen.add(eid)
                    unique_rows.append(row)
            
            cands: List[Candidate] = []
            for row in unique_rows:
                depth = int(row["depth"])
                ent_ranges = _parse_int_list(row.get("ent_ranges", ""), depth, 1)
                cnot_modes = _parse_cnot_modes(row.get("cnot_modes", ""), depth, cnot_lookup)
                base_lr = float(row["learning_rate"])
                final_lr = float(lr_override) if lr_override is not None else base_lr * lr_scale
                cands.append(Candidate(
                    eval_id=str(row["eval_id"]),
                    generation=int(row["generation"]),
                    embed=str(row["embed"]).lower(),
                    n_qubits=int(row["n_qubits"]),
                    depth=depth,
                    ent_ranges=ent_ranges,
                    cnot_modes=cnot_modes,
                    learning_rate=final_lr,
                    shots=int(shots),
                    selection=reasons[row["eval_id"]],
                    pre_val_acc=float(row["val_acc"]),
                    pre_val_loss=float(row.get("val_loss", np.nan)),
                    original_learning_rate=base_lr,
                    meta={
                        "f2_circuit_cost": float(row.get("f2_circuit_cost", np.nan)),
                        "f3_n_subcircuits": float(row.get("f3_n_subcircuits", np.nan)),
                        "seconds": float(row.get("seconds", np.nan)),
                    },
                ))

    print(f"[SELECTION] Created {len(cands)} candidate configurations")
    
    # Try to load and include the "final-best" model if it exists in the training log
    if epoch_df is not None:
        final_best_entries = epoch_df[epoch_df['eval_id'] == 'final-best'].copy()
        if not final_best_entries.empty:
            # Get the final epoch validation results
            final_best_val = final_best_entries[final_best_entries['val_acc'].notna()]
            if not final_best_val.empty:
                # Get the last epoch (highest epoch number)
                final_best_final = final_best_val.sort_values('epoch').iloc[-1]
                final_val_acc = final_best_final['val_acc']
                final_val_loss = final_best_final.get('val_loss', np.nan)
                
                # Get architecture from the final-best entry
                embed = str(final_best_final.get('embed', 'angle-x')).lower()
                n_qubits = int(final_best_final.get('n_qubits', 9))
                depth = int(final_best_final.get('depth', 2))
                ent_ranges_str = str(final_best_final.get('ent_ranges', ''))
                cnot_modes_str = str(final_best_final.get('cnot_modes', ''))
                base_lr = float(final_best_final.get('lr', 0.004369619))
                
                ent_ranges = _parse_int_list(ent_ranges_str, depth, 1)
                cnot_modes = _parse_cnot_modes(cnot_modes_str, depth, cnot_lookup)
                final_lr = float(lr_override) if lr_override is not None else base_lr * lr_scale
                
                # Find the corresponding NSGA-II eval that this final-best came from
                # The final-best should match the best overall model from NSGA-II
                best_nsga_eval = evals.loc[evals["val_acc"].idxmax()]
                nsga_eval_id = best_nsga_eval["eval_id"]
                nsga_generation = int(best_nsga_eval["generation"])
                
                # Create a candidate for the final-best model
                # Use the NSGA-II pre-training accuracy as baseline
                final_best_candidate = Candidate(
                    eval_id="final-best",
                    generation=nsga_generation,
                    embed=embed,
                    n_qubits=n_qubits,
                    depth=depth,
                    ent_ranges=ent_ranges,
                    cnot_modes=cnot_modes,
                    learning_rate=final_lr,
                    shots=int(shots),
                    selection="final_best",
                    pre_val_acc=float(best_nsga_eval["val_acc"]),  # Use NSGA-II accuracy as baseline
                    pre_val_loss=float(best_nsga_eval.get("val_loss", np.nan)),
                    original_learning_rate=base_lr,
                    meta={
                        "f2_circuit_cost": float(best_nsga_eval.get("f2_circuit_cost", np.nan)),
                        "f3_n_subcircuits": float(best_nsga_eval.get("f3_n_subcircuits", np.nan)),
                        "seconds": float(best_nsga_eval.get("seconds", np.nan)),
                        "final_best_val_acc": float(final_val_acc),
                        "final_best_epochs": int(final_best_final['epoch']),
                        "original_eval_id": nsga_eval_id,
                    },
                )
                
                cands.append(final_best_candidate)
                print(f"[SELECTION] Added final-best model (trained from {nsga_eval_id})")
                print(f"[SELECTION]   NSGA-II accuracy: {best_nsga_eval['val_acc']:.2f}%")
                print(f"[SELECTION]   Final-best accuracy: {final_val_acc:.2f}% (after {int(final_best_final['epoch'])} epochs)")
        else:
            print(f"[SELECTION] No 'final-best' model found in training logs")
    
    # Ensure parameter diversity
    diverse_cands = _ensure_parameter_diversity(cands)
    
    # Analyze and report diversity metrics
    _analyze_sample_diversity(diverse_cands)
    
    return diverse_cands


# --------------------------
# Orchestration (2 workers/GPU, Main-style logging)
# --------------------------

def _import_hq_module(name: str):
    """Import or reload the HQ module to pick up environment variable changes."""
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    # Reload the module if it's already imported to pick up new environment variables
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    else:
        return importlib.import_module(name)

def _ensure_module_environment(dataset: Optional[str], run_dir: Path) -> None:
    # Tell Main.py to not create folders when imported as a module
    os.environ["IMPORTED_AS_MODULE"] = "true"
    os.environ["RUN_TYPE"] = "correlation"  # Mark this as a correlation analysis run
    if dataset: os.environ.setdefault("DATASET", dataset.lower())
    # CRITICAL: Override LOG_DIR to point to run_dir to prevent files being created in original logs directory
    os.environ["LOG_DIR"] = str(run_dir)
    # Use Main-like filenames in the analysis run directory (force override, don't use setdefault)
    os.environ["EPOCH_LOG_CSV"] = str(run_dir / "train_epoch_log.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(run_dir / "checkpoint_validation.csv")
    # Set progress log to the run directory (not the dataset log directory)
    os.environ["PROGRESS_LOG"] = str(run_dir / "progress.log")
    # Set these to non-existent paths to prevent Main.py from creating NSGA-II files
    os.environ["NSGA_EVAL_CSV"] = "/dev/null"
    os.environ["GEN_SUMMARY_CSV"] = "/dev/null"
    
    # Enable checkpoint validation for correlation analysis
    # Load checkpoint settings from environment (already set by .env)
    checkpoint_enabled = os.getenv("CHECKPOINT_CORRELATION_ENABLED", "true").lower() in ("true", "1", "yes")
    if checkpoint_enabled:
        os.environ.setdefault("CHECKPOINT_VALIDATION_ENABLED", "true")
        # Pass through checkpoint configuration if not already set
        if "CHECKPOINT_TRAIN_SIZES" not in os.environ:
            os.environ["CHECKPOINT_TRAIN_SIZES"] = "512,1024,2048,4096,8196,16392,32768,full"
        if "CHECKPOINT_TARGET_EPOCHS" not in os.environ:
            os.environ["CHECKPOINT_TARGET_EPOCHS"] = "1,3,5,10"

def _worker_wrapper(job, args_dict, run_dir, gpu_id, worker_rank, hq_module_name, lock_proxy, result_queue):
    """Wrapper function for multiprocessing that calls _run_single_retrain and puts result in queue"""
    try:
        result = _run_single_retrain(job, args_dict, run_dir, gpu_id, worker_rank, hq_module_name, lock_proxy)
        result_queue.put(result)
    except Exception as e:
        # If something goes wrong, put an error result
        result_queue.put(RetrainResult(
            eval_id=str(job["eval_id"]),
            post_val_acc=None,
            post_val_loss=None,
            epochs=int(args_dict["final_epochs"]),
            lr_used=float(job["learning_rate"]),
            shots_used=int(job["shots"]),
            device=f"cuda:{gpu_id}" if gpu_id is not None else "cpu",
            seconds=None,
            weights_path=None,
            error=f"Worker wrapper error: {str(e)}"
        ))


def _run_single_retrain(
    job: Dict[str, object],
    args_dict: Dict[str, object],
    run_dir: Path,
    gpu_id: Optional[int],
    worker_rank: int,
    hq_module_name: str,
    lock_proxy,  # Manager RLock passed from parent so _csv_append/_append_progress are serialized
) -> RetrainResult:
    import torch

    # Isolate device
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Single-threaded DataLoader path inside training module (like pool worker)
    os.environ["QNAS_POOL_WORKER"] = "1"

    # Ensure environment is set before importing module
    _ensure_module_environment(None, run_dir)

    HQMain = _import_hq_module(hq_module_name)

    # Share the CSV lock so logging is protected across processes
    HQMain.CSV_LOCK = lock_proxy

    # Force override all log paths to run_dir to prevent writing to original logs directory
    HQMain.LOG_DIR = str(run_dir)
    HQMain.DATASET_LOG_DIR = str(run_dir)
    HQMain.EPOCH_LOG_CSV = str(run_dir / "train_epoch_log.csv")
    HQMain.CHECKPOINT_LOG_CSV = str(run_dir / "checkpoint_validation.csv")
    HQMain.PROGRESS_LOG = str(run_dir / "progress.log")
    
    # Also update logging_utils module if it's imported
    try:
        from qnas.utils import logging_utils as lu
        lu.DATASET_LOG_DIR = str(run_dir)
        lu.EPOCH_LOG_CSV = str(run_dir / "train_epoch_log.csv")
        lu.CHECKPOINT_LOG_CSV = str(run_dir / "checkpoint_validation.csv")
        lu.PROGRESS_LOG = str(run_dir / "progress.log")
    except (ImportError, AttributeError):
        pass  # logging_utils might not be accessible this way
    
    # Also update config module
    try:
        from qnas.utils import config as cfg
        cfg.LOG_DIR = str(run_dir)
        cfg.DATASET_LOG_DIR = str(run_dir)
    except (ImportError, AttributeError):
        pass
    # Ensure headers exist (Main's helper)
    HQMain._csv_prepare(HQMain.EPOCH_LOG_CSV, HQMain.EPOCH_HEADER)

    # Cosmetics for logs
    HQMain.WORKER_GPU_ID = gpu_id if gpu_id is not None else -1
    HQMain.WORKER_RANK   = worker_rank

    shots_used = int(job["shots"])
    
    # Set training seed if provided for diverse sampling
    train_seed = job.get("train_seed")
    if train_seed is not None:
        print(f"[GPU{gpu_id}|{job['eval_id']}] Using train seed: {train_seed}")
        # Set the seed in the environment for Main.py to pick up
        os.environ["TRAIN_SEED"] = str(train_seed)
    
    cfg = HQMain.QConfig(
        embed_kind=str(job["embed"]),
        n_qubits=int(job["n_qubits"]),
        depth=int(job["depth"]),
        ent_ranges=list(job["ent_ranges"]),
        cnot_modes=list(job["cnot_modes"]),
        learning_rate=float(job["learning_rate"]),
        shots=shots_used,
    )

    epochs    = int(args_dict["final_epochs"])
    train_sz  = args_dict.get("final_train_size")
    val_sz    = args_dict.get("final_val_size")
    save_wts  = bool(args_dict.get("save_weights"))

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = f"cuda:{torch.cuda.current_device()}"
        # Clear any existing memory before starting
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    t0 = time.time()
    try:
        vloss, vacc, model, _, _ = HQMain.train_for_budget(
            cfg,
            eval_id=f"final-{job['eval_id']}",
            epochs=epochs,
            max_train_batches=0,
            max_val_batches=0,
            train_size=train_sz,
            val_size=val_sz,
        )
        seconds = time.time() - t0

        weights_path = None
        if save_wts:
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            weights_path = run_dir / "weights" / f"{job['eval_id']}_final.pt"
            HQMain.save_model_weights(model, str(weights_path), cfg, eval_id=f"final-{job['eval_id']}", epoch=epochs, val_acc=vacc, val_loss=vloss)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return RetrainResult(
            eval_id=str(job["eval_id"]),
            post_val_acc=float(vacc),
            post_val_loss=float(vloss),
            epochs=epochs,
            lr_used=float(job["learning_rate"]),
            shots_used=shots_used,
            device=device,
            seconds=seconds,
            weights_path=str(weights_path) if weights_path else None,
        )
    except KeyboardInterrupt:
        # Handle interruption gracefully
        error_msg = "Training interrupted by user (KeyboardInterrupt)"
        print(f"[{job['eval_id']}] [INTERRUPTED] {error_msg}")
        
        # Clean up GPU memory after interruption
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        
        with open(run_dir / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {job['eval_id']} interrupted:\n")
            f.write(f"Config: embed={job['embed']}, n_qubits={job['n_qubits']}, depth={job['depth']}\n")
            f.write(f"{traceback.format_exc()}\n")
            f.write("NOTE: Training was interrupted. This may be due to manual cancellation or system resource limits.\n\n")
        
        return RetrainResult(
            eval_id=str(job["eval_id"]),
            post_val_acc=None,
            post_val_loss=None,
            epochs=epochs,
            lr_used=float(job["learning_rate"]),
            shots_used=shots_used,
            device=device,
            seconds=None,
            weights_path=None,
            error=error_msg,
        )
    except Exception as exc:
        error_msg = str(exc)
        exc_type = type(exc).__name__
        
        # Clean up GPU memory after error
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        
        # Check if this is a GPU memory error
        is_memory_error = (
            "CUDA" in error_msg or 
            "out of memory" in error_msg.lower() or 
            "custatevec" in error_msg.lower() or
            "memory allocation" in error_msg.lower()
        )
        
        error_prefix = "[GPU_MEMORY_ERROR]" if is_memory_error else "[ERROR]"
        detailed_error = f"{error_prefix} {exc_type}: {error_msg}"
        
        with open(run_dir / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {job['eval_id']} failed:\n")
            f.write(f"Config: embed={job['embed']}, n_qubits={job['n_qubits']}, depth={job['depth']}\n")
            f.write(f"{traceback.format_exc()}\n")
            if is_memory_error:
                f.write("NOTE: This appears to be a GPU memory allocation failure. "
                       "Consider reducing batch size or using a smaller model configuration.\n\n")
        
        print(f"[{job['eval_id']}] {detailed_error}")
        
        return RetrainResult(
            eval_id=str(job["eval_id"]),
            post_val_acc=None,
            post_val_loss=None,
            epochs=epochs,
            lr_used=float(job["learning_rate"]),
            shots_used=shots_used,
            device=device,
            seconds=None,
            weights_path=None,
            error=str(exc),
        )

def _candidate_to_job(c: Candidate, train_seed: Optional[int] = None) -> Dict[str, object]:
    job = {
        "eval_id": c.eval_id,
        "generation": c.generation,
        "embed": c.embed,
        "n_qubits": c.n_qubits,
        "depth": c.depth,
        "ent_ranges": c.ent_ranges,
        "cnot_modes": c.cnot_modes,
        "learning_rate": c.learning_rate,
        "shots": c.shots,
        "selection": c.selection,
        "pre_val_acc": c.pre_val_acc,
        "pre_val_loss": c.pre_val_loss,
        "original_learning_rate": c.original_learning_rate,
        "meta": c.meta,
    }
    if train_seed is not None:
        job["train_seed"] = train_seed
    return job

def retrain_candidates(
    candidates: List[Candidate],
    args: argparse.Namespace,
    run_dir: Path,
    dataset: Optional[str],
) -> List[RetrainResult]:
    # Create jobs with diverse seeds if requested
    if args.diverse_seeds:
        print(f"[DIVERSITY] Using diverse seeds for {len(candidates)} training jobs")
        # Generate different seeds for each job based on base seed
        base_rng = np.random.default_rng(args.seed)
        train_seeds = base_rng.integers(0, 1000000, size=len(candidates))
        jobs = [_candidate_to_job(c, seed) for c, seed in zip(candidates, train_seeds)]
        
        # Log the seed assignments for reproducibility
        seed_log = run_dir / "train_seeds.json"
        seed_data = {job["eval_id"]: job["train_seed"] for job in jobs}
        with open(seed_log, 'w') as f:
            json.dump(seed_data, f, indent=2)
        print(f"[DIVERSITY] Train seeds saved to {seed_log}")
    else:
        jobs = [_candidate_to_job(c) for c in candidates]
    
    if not jobs:
        return []

    import torch
    gpus_available = list(range(torch.cuda.device_count()))
    use_all = (args.max_gpus is None) or (args.max_gpus == 0)
    max_gpus = len(gpus_available) if use_all else min(args.max_gpus, len(gpus_available))
    selected_gpus = gpus_available[:max_gpus]

    args_dict = {
        "final_epochs": int(args.final_epochs),
        "final_shots": int(args.final_shots),
        "final_train_size": args.train_size,
        "final_val_size": args.val_size,
        "save_weights": args.save_weights,
        "final_lr": args.final_lr,
    }

    _ensure_module_environment(dataset, run_dir)

    # CPU fallback: sequential, in-process
    if not selected_gpus:
        print("[INFO] No GPUs detected or allowed; running on CPU.")
        # A dummy lock for single-process mode
        class _DummyLock: 
            def __enter__(self): return None
            def __exit__(self, *a): return False
        results: List[RetrainResult] = []
        for idx, job in enumerate(jobs):
            results.append(
                _run_single_retrain(job, args_dict, run_dir, gpu_id=None,
                                    worker_rank=idx, hq_module_name=args.hq_module,
                                    lock_proxy=_DummyLock())
            )
        return results

    print(f"[INFO] Using GPUs {selected_gpus} with {args.workers_per_gpu} workers/GPU (one QNAS model per worker).")

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    lock_proxy = manager.RLock()  # shared CSV/PROGRESS lock across all workers
    result_queue: "mp.Queue" = ctx.Queue()

    # Slot accounting: two workers per GPU by default
    slots_free = {gid: int(args.workers_per_gpu) for gid in selected_gpus}

    def next_free_gpu() -> Optional[int]:
        for gid in selected_gpus:
            if slots_free[gid] > 0:
                return gid
        return None

    pending_by_pid: Dict[int, str] = {}          # pid -> eval_id
    active: Dict[int, Tuple[mp.Process, int]] = {}  # pid -> (proc, gpu_id)

    results: List[RetrainResult] = []
    next_job = 0
    total_jobs = len(jobs)
    jobs_completed = 0
    worker_rank_counter = 0

    def launch(gpu_id: int, job_idx: int):
        nonlocal worker_rank_counter
        job = jobs[job_idx]
        wrank = worker_rank_counter
        worker_rank_counter += 1
        # Child computes then pushes a RetrainResult into the queue
        p = ctx.Process(
            target=_worker_wrapper,
            args=(job, args_dict, run_dir, gpu_id, wrank, args.hq_module, lock_proxy, result_queue),
            name=f"retrain-gpu{gpu_id}-job{job_idx}",
        )
        p.start()
        pending_by_pid[p.pid] = job["eval_id"]
        active[p.pid] = (p, gpu_id)
        slots_free[gpu_id] -= 1

    # Fill available slots
    while next_job < total_jobs:
        gid = next_free_gpu()
        if gid is None: break
        launch(gid, next_job)
        next_job += 1

    # Event loop
    while jobs_completed < total_jobs:
        try:
            res = result_queue.get(timeout=5.0)
            if isinstance(res, RetrainResult):
                results.append(res)
                jobs_completed += 1
        except queue.Empty:
            pass

        # Reclaim finished/crashed, free slots, and launch new jobs
        finished_pids: List[int] = []
        for pid, (proc, gid) in list(active.items()):
            if not proc.is_alive():
                proc.join()
                finished_pids.append(pid)
                slots_free[gid] += 1
                eval_id = pending_by_pid.get(pid)
                # If no result arrived for this eval_id, synthesize a failure
                if eval_id is not None and not any(r.eval_id == eval_id for r in results):
                    results.append(RetrainResult(
                        eval_id=eval_id, post_val_acc=None, post_val_loss=None,
                        epochs=int(args.final_epochs), lr_used=float(args.final_lr or 0.0),
                        shots_used=int(args.final_shots), device=f"cuda:{gid}",
                        seconds=None, weights_path=None,
                        error="Worker exited without returning a result."
                    ))
                    jobs_completed += 1
                del active[pid]; pending_by_pid.pop(pid, None)

        while next_job < total_jobs:
            gid = next_free_gpu()
            if gid is None: break
            launch(gid, next_job)
            next_job += 1

    # Ensure no stray processes
    for pid, (proc, _) in list(active.items()):
        if proc.is_alive(): proc.join()
    
    # Properly shutdown the Manager to release semaphores
    manager.shutdown()

    return results


# --------------------------
# Analytics & plotting
# --------------------------

def correlate(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "order_consistency": float("nan")}
    pre = df["pre_val_acc"].to_numpy()
    post = df["post_val_acc"].to_numpy()
    pearson = float(np.corrcoef(pre, post)[0, 1])
    pre_rank = pd.Series(pre).rank(method="average").to_numpy()
    post_rank = pd.Series(post).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(pre_rank, post_rank)[0, 1])
    consistent = 0; comparable = 0
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            dp = pre[i] - pre[j]; dq = post[i] - post[j]
            if dp == 0 or dq == 0: continue
            comparable += 1
            if dp * dq > 0: consistent += 1
    return {"pearson": pearson, "spearman": spearman, "order_consistency": (consistent / comparable) if comparable else float("nan")}

def plot_scatter(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    palette = {"best_generation": "#1f77b4", "random_generation": "#ff7f0e", "random_global": "#2ca02c"}
    colors = df["selection"].map(palette).fillna("#7f7f7f")
    plt.scatter(df["pre_val_acc"], df["post_val_acc"], c=colors, s=60, edgecolors="k", alpha=0.85)
    vmin = float(min(df["pre_val_acc"].min(), df["post_val_acc"].min()))
    vmax = float(max(df["pre_val_acc"].max(), df["post_val_acc"].max()))
    pad = max(0.5, 0.02 * (vmax - vmin))
    lims = [vmin - pad, vmax + pad]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims); plt.ylim(lims); plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("NSGA-II validation accuracy (%)"); plt.ylabel("Final retrain accuracy (%)")
    plt.title("Before vs After accuracy")
    handles = []
    for key, color in palette.items():
        if key in df["selection"].values:
            handles.append(plt.Line2D([0], [0], marker="o", color="w", label=key.replace("_", " "),
                                      markerfacecolor=color, markeredgecolor="k", markersize=9))
    if handles: plt.legend(handles=handles, title="Selection", loc="lower right")
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def plot_trend(df: pd.DataFrame, path: Path) -> None:
    best = df[df["selection"] == "best_generation"].sort_values("generation")
    if best.empty: return
    plt.figure(figsize=(8, 5))
    plt.plot(best["generation"], best["pre_val_acc"], "o-", label="NSGA-II best", linewidth=2)
    plt.plot(best["generation"], best["post_val_acc"], "s-", label="Final retrain", linewidth=2)
    plt.xlabel("Generation"); plt.ylabel("Validation accuracy (%)")
    plt.title("Best-per-generation accuracy trend")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


# --------------------------
# Epoch-by-epoch correlation tracking
# --------------------------

def extract_epoch_correlation(run_dir: Path, output_csv: Optional[Path] = None, epochs_filter: Optional[List[int]] = None) -> pd.DataFrame:
    """Extract epoch-by-epoch correlation between NSGA-II accuracy and final training accuracy.
    
    Args:
        run_dir: Path to the run directory (e.g., logs/mnist_2025-09-25/run_20251006-163458)
        output_csv: Optional path to save the output CSV. If None, saves to run_dir/epoch_correlation.csv
        epochs_filter: Optional list of specific epochs to extract (e.g., [1, 5, 10, 20]). If None, extracts all epochs.
    
    Returns:
        DataFrame with columns: eval_id, epoch, pre_val_acc, val_acc, delta_acc, embed, n_qubits, depth, etc.
    """
    run_dir = Path(run_dir)
    
    # Print filter information if provided
    if epochs_filter:
        print(f"[INFO] Filtering to specific epochs: {epochs_filter}")
    
    # Load the selected candidates file to get pre-training accuracy
    selected_candidates_path = run_dir / "selected_candidates.csv"
    if not selected_candidates_path.exists():
        raise FileNotFoundError(f"selected_candidates.csv not found in {run_dir}")
    
    selected_df = pd.read_csv(selected_candidates_path)
    # Strip whitespace from column names and string values
    selected_df.columns = selected_df.columns.str.strip()
    for col in selected_df.select_dtypes(include=['object']).columns:
        selected_df[col] = selected_df[col].str.strip()
    
    # Load the NSGA-II results to find the best NSGA-II candidate
    nsga_dir = run_dir.parent  # Go up to logs/mnist_2025-09-25/
    nsga_evals_path = nsga_dir / "nsga_evals.csv"
    nsga_train_log_path = nsga_dir / "train_epoch_log.csv"
    nsga_best_candidate = None
    nsga_best_epochs = None
    
    if nsga_evals_path.exists():
        nsga_df = pd.read_csv(nsga_evals_path)
        nsga_df.columns = nsga_df.columns.str.strip()
        for col in nsga_df.select_dtypes(include=['object']).columns:
            nsga_df[col] = nsga_df[col].str.strip()
        
        # Find the best NSGA-II candidate (highest val_acc)
        if not nsga_df.empty and 'val_acc' in nsga_df.columns:
            best_nsga_idx = nsga_df['val_acc'].idxmax()
            nsga_best_candidate = nsga_df.loc[best_nsga_idx].to_dict()
            print(f"[INFO] Best NSGA-II candidate: {nsga_best_candidate['eval_id']} with accuracy: {nsga_best_candidate['val_acc']:.2f}%")
            
            # Load NSGA-II training epochs for this candidate
            if nsga_train_log_path.exists():
                nsga_train_df = pd.read_csv(nsga_train_log_path)
                nsga_train_df.columns = nsga_train_df.columns.str.strip()
                for col in nsga_train_df.select_dtypes(include=['object']).columns:
                    nsga_train_df[col] = nsga_train_df[col].str.strip()
                
                # Get epoch-end rows for the best NSGA-II candidate
                nsga_best_epochs = nsga_train_df[
                    (nsga_train_df['eval_id'] == nsga_best_candidate['eval_id']) &
                    (nsga_train_df['val_acc'].notna())
                ].copy()
                
                if not nsga_best_epochs.empty:
                    nsga_best_epochs = nsga_best_epochs.sort_values('epoch')
                    print(f"[INFO] Found {len(nsga_best_epochs)} training epochs for best NSGA-II candidate")
    else:
        print(f"[WARNING] NSGA-II evals file not found at {nsga_evals_path}")
    
    # Load the epoch training log
    epoch_log_path = run_dir / "train_epoch_log.csv"
    if not epoch_log_path.exists():
        raise FileNotFoundError(f"train_epoch_log.csv not found in {run_dir}")
    
    epoch_df = pd.read_csv(epoch_log_path)
    # Strip whitespace from column names and string values
    epoch_df.columns = epoch_df.columns.str.strip()
    for col in epoch_df.select_dtypes(include=['object']).columns:
        epoch_df[col] = epoch_df[col].str.strip()
    
    # Filter to only validation phase (end of epoch)
    # The validation rows have val_acc populated and phase might be 'val' or similar
    val_epochs = epoch_df[epoch_df['val_acc'].notna()].copy()
    
    if val_epochs.empty:
        print(f"[WARNING] No validation accuracy data found in {epoch_log_path}")
        return pd.DataFrame()
    
    # Extract the original eval_id from the final eval_id (remove "final-" prefix)
    val_epochs['original_eval_id'] = val_epochs['eval_id'].str.replace('final-', '', regex=False)
    
    # Determine the final best candidate (highest final accuracy across all epochs)
    # Group by eval_id and get max epoch for each
    final_epoch_per_candidate = val_epochs.groupby('original_eval_id')['epoch'].max().reset_index()
    final_epoch_data = val_epochs.merge(final_epoch_per_candidate, 
                                        left_on=['original_eval_id', 'epoch'],
                                        right_on=['original_eval_id', 'epoch'])
    
    # Find the candidate with highest final validation accuracy
    if not final_epoch_data.empty:
        best_final_idx = final_epoch_data['val_acc'].idxmax()
        best_final_eval_id = final_epoch_data.loc[best_final_idx, 'original_eval_id']
        best_final_acc = final_epoch_data.loc[best_final_idx, 'val_acc']
        print(f"[INFO] Final best candidate: {best_final_eval_id} with final accuracy: {best_final_acc:.2f}%")
    else:
        best_final_eval_id = None
    
    # Merge with selected candidates to get pre_val_acc and objectives
    correlation_data = []
    
    for _, candidate in selected_df.iterrows():
        eval_id = candidate['eval_id']
        pre_val_acc = candidate['pre_val_acc']
        pre_val_loss = candidate['pre_val_loss']
        generation = candidate['generation']
        selection = candidate['selection']
        
        # Calculate f1 from pre_val_acc
        f1_pre = 100.0 - pre_val_acc  # f1 = 1 - accuracy (as percentage)
        
        # Get f2 and f3 from candidate data
        f2 = candidate.get('f2_circuit_cost', None)
        f3 = candidate.get('f3_n_subcircuits', None)
        
        # Create descriptive label based on selection type
        if selection == 'best_generation':
            label = f'best_gen_{generation}'
        elif selection == 'random_generation':
            label = f'random_gen_{generation}'
        else:
            label = f'{selection}_gen_{generation}'
        
        # Find all epochs for this eval_id
        eval_epochs = val_epochs[val_epochs['original_eval_id'] == eval_id].copy()
        
        if eval_epochs.empty:
            print(f"[WARNING] No training data found for {eval_id}")
            continue
        
        # Sort by epoch to ensure proper ordering
        eval_epochs = eval_epochs.sort_values('epoch')
        
        # Filter to specific epochs if requested
        if epochs_filter:
            eval_epochs = eval_epochs[eval_epochs['epoch'].isin(epochs_filter)]
            if eval_epochs.empty:
                print(f"[WARNING] No data found for {eval_id} at epochs {epochs_filter}")
                continue
        
        for _, epoch_row in eval_epochs.iterrows():
            # Calculate f1 for this epoch
            f1_epoch = 100.0 - epoch_row['val_acc']  # f1 = 1 - accuracy (as percentage)
            
            correlation_data.append({
                'eval_id': eval_id,
                'epoch': epoch_row['epoch'],
                'label': label,
                'pre_val_acc': pre_val_acc,
                'val_acc': epoch_row['val_acc'],
                'delta_acc': epoch_row['val_acc'] - pre_val_acc,
                'f1_pre': f1_pre,
                'f1_epoch': f1_epoch,
                'f1_delta': f1_epoch - f1_pre,
                'f2_circuit_cost': f2,
                'f3_n_subcircuits': f3,
                'pre_val_loss': pre_val_loss,
                'val_loss': epoch_row['val_loss'],
                'delta_loss': epoch_row['val_loss'] - pre_val_loss,
                'train_acc': epoch_row.get('train_acc', None),
                'train_loss': epoch_row.get('train_loss', None),
                'embed': candidate['embed'],
                'n_qubits': candidate['n_qubits'],
                'depth': candidate['depth'],
                'ent_ranges': candidate['ent_ranges'],
                'cnot_modes': candidate['cnot_modes'],
                'generation': candidate['generation'],
                'learning_rate': epoch_row.get('lr', candidate.get('original_learning_rate', None)),
                'gpu_id': epoch_row.get('gpu_id', None),
                'elapsed_s': epoch_row.get('elapsed_s', None),
            })
    
    # Add the best NSGA-II candidate if it wasn't already selected for retraining
    if nsga_best_candidate is not None:
        best_nsga_eval_id = nsga_best_candidate['eval_id']
        if best_nsga_eval_id not in selected_df['eval_id'].values:
            print(f"[INFO] Adding best NSGA-II candidate {best_nsga_eval_id} to correlation data (not retrained)")
            
            # Parse entanglement ranges and cnot modes
            depth = int(nsga_best_candidate.get('depth', 0))
            ent_ranges_str = str(nsga_best_candidate.get('ent_ranges', ''))
            cnot_modes_str = str(nsga_best_candidate.get('cnot_modes', ''))
            generation = nsga_best_candidate.get('gen_est', 0)
            
            # Check if the best NSGA-II candidate was retrained as "final-best"
            # Look for "final-best" in the parent NSGA-II training log
            final_best_epochs = None
            if nsga_train_log_path.exists():
                # Look for "final-best" entries in NSGA-II training log
                final_best_epochs = nsga_train_df[
                    (nsga_train_df['eval_id'] == 'final-best') &
                    (nsga_train_df['val_acc'].notna())
                ].copy()
                
                if not final_best_epochs.empty:
                    final_best_epochs = final_best_epochs.sort_values('epoch')
                    print(f"[INFO] Found 'final-best' retraining data for best NSGA-II candidate ({len(final_best_epochs)} epochs)")
            
            # Use final-best retraining epochs if available, otherwise use NSGA-II epochs
            if final_best_epochs is not None and not final_best_epochs.empty:
                # Use the final NSGA-II accuracy as pre_val_acc
                final_nsga_acc = nsga_best_candidate['val_acc']
                final_nsga_loss = nsga_best_candidate.get('val_loss', np.nan)
                
                # Use the final-best retraining epochs (should be 1, 2, 3)
                for _, epoch_row in final_best_epochs.iterrows():
                    epoch_val_acc = epoch_row['val_acc']
                    f1_pre = 100.0 - final_nsga_acc
                    f1_epoch = 100.0 - epoch_val_acc
                    
                    correlation_data.append({
                        'eval_id': best_nsga_eval_id,
                        'epoch': int(epoch_row['epoch']),
                        'label': f'nsga_best_gen_{generation}',
                        'pre_val_acc': final_nsga_acc,  # Use final NSGA-II accuracy as baseline
                        'val_acc': epoch_val_acc,
                        'delta_acc': epoch_val_acc - final_nsga_acc,
                        'f1_pre': f1_pre,
                        'f1_epoch': f1_epoch,
                        'f1_delta': f1_epoch - f1_pre,
                        'f2_circuit_cost': nsga_best_candidate.get('f2_circuit_cost', None),
                        'f3_n_subcircuits': nsga_best_candidate.get('f3_n_subcircuits', None),
                        'pre_val_loss': final_nsga_loss,
                        'val_loss': epoch_row['val_loss'],
                        'delta_loss': epoch_row['val_loss'] - final_nsga_loss,
                        'train_acc': epoch_row.get('train_acc', None),
                        'train_loss': epoch_row.get('train_loss', None),
                        'embed': nsga_best_candidate.get('embed', ''),
                        'n_qubits': nsga_best_candidate.get('n_qubits', 0),
                        'depth': depth,
                        'ent_ranges': ent_ranges_str,
                        'cnot_modes': cnot_modes_str,
                        'generation': generation,
                        'learning_rate': nsga_best_candidate.get('learning_rate', None),
                        'gpu_id': epoch_row.get('gpu_id', None),
                        'elapsed_s': epoch_row.get('elapsed_s', None),
                    })
            elif nsga_best_epochs is not None and not nsga_best_epochs.empty:
                # Fallback: Use NSGA-II training epochs if no final-best retraining found
                print(f"[WARNING] No 'final-best' retraining found, using NSGA-II epochs")
                final_nsga_acc = nsga_best_candidate['val_acc']
                final_nsga_loss = nsga_best_candidate.get('val_loss', np.nan)
                
                epochs_to_use = nsga_best_epochs.head(3)
                for idx, (_, epoch_row) in enumerate(epochs_to_use.iterrows(), 1):
                    epoch_val_acc = epoch_row['val_acc']
                    f1_pre = 100.0 - final_nsga_acc
                    f1_epoch = 100.0 - epoch_val_acc
                    
                    correlation_data.append({
                        'eval_id': best_nsga_eval_id,
                        'epoch': idx,
                        'label': f'nsga_best_gen_{generation}',
                        'pre_val_acc': final_nsga_acc,
                        'val_acc': epoch_val_acc,
                        'delta_acc': epoch_val_acc - final_nsga_acc,
                        'f1_pre': f1_pre,
                        'f1_epoch': f1_epoch,
                        'f1_delta': f1_epoch - f1_pre,
                        'f2_circuit_cost': nsga_best_candidate.get('f2_circuit_cost', None),
                        'f3_n_subcircuits': nsga_best_candidate.get('f3_n_subcircuits', None),
                        'pre_val_loss': final_nsga_loss,
                        'val_loss': epoch_row['val_loss'],
                        'delta_loss': epoch_row['val_loss'] - final_nsga_loss,
                        'train_acc': epoch_row.get('train_acc', None),
                        'train_loss': epoch_row.get('train_loss', None),
                        'embed': nsga_best_candidate.get('embed', ''),
                        'n_qubits': nsga_best_candidate.get('n_qubits', 0),
                        'depth': depth,
                        'ent_ranges': ent_ranges_str,
                        'cnot_modes': cnot_modes_str,
                        'generation': generation,
                        'learning_rate': nsga_best_candidate.get('learning_rate', None),
                        'gpu_id': epoch_row.get('gpu_id', None),
                        'elapsed_s': epoch_row.get('elapsed_s', None),
                    })
    
    result_df = pd.DataFrame(correlation_data)
    
    if result_df.empty:
        print(f"[WARNING] No correlation data could be extracted from {run_dir}")
        return result_df
    
    # Save to CSV
    if output_csv is None:
        output_csv = run_dir / "epoch_correlation.csv"
    else:
        output_csv = Path(output_csv)
    
    result_df.to_csv(output_csv, index=False)
    print(f"[INFO] Epoch correlation data saved to {output_csv}")
    print(f"[INFO] Extracted data for {result_df['eval_id'].nunique()} candidates across {result_df['epoch'].max()} epochs")
    
    return result_df


def analyze_epoch_correlation(correlation_df: pd.DataFrame, run_dir: Path) -> Dict[str, float]:
    """Analyze the correlation between pre-training and post-training accuracy across epochs.
    
    Args:
        correlation_df: DataFrame from extract_epoch_correlation
        run_dir: Path to save analysis plots
    
    Returns:
        Dictionary with correlation statistics per epoch
    """
    if correlation_df.empty:
        return {}
    
    run_dir = Path(run_dir)
    
    # Calculate correlation for each epoch
    epoch_stats = []
    for epoch in sorted(correlation_df['epoch'].unique()):
        epoch_data = correlation_df[correlation_df['epoch'] == epoch]
        
        if len(epoch_data) < 2:
            continue
        
        pre_acc = epoch_data['pre_val_acc'].to_numpy()
        post_acc = epoch_data['val_acc'].to_numpy()
        
        # Pearson correlation
        pearson = np.corrcoef(pre_acc, post_acc)[0, 1] if len(pre_acc) > 1 else np.nan
        
        # Spearman (rank) correlation
        pre_rank = pd.Series(pre_acc).rank(method="average").to_numpy()
        post_rank = pd.Series(post_acc).rank(method="average").to_numpy()
        spearman = np.corrcoef(pre_rank, post_rank)[0, 1] if len(pre_rank) > 1 else np.nan
        
        epoch_stats.append({
            'epoch': epoch,
            'pearson': pearson,
            'spearman': spearman,
            'mean_pre_acc': pre_acc.mean(),
            'mean_post_acc': post_acc.mean(),
            'mean_delta_acc': (post_acc - pre_acc).mean(),
            'n_samples': len(epoch_data),
        })
    
    stats_df = pd.DataFrame(epoch_stats)
    stats_df.to_csv(run_dir / "epoch_correlation_stats.csv", index=False)
    
    # Create visualization
    if len(stats_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Correlation over epochs
        ax = axes[0, 0]
        ax.plot(stats_df['epoch'], stats_df['pearson'], 'o-', label='Pearson', linewidth=2, markersize=8)
        ax.plot(stats_df['epoch'], stats_df['spearman'], 's-', label='Spearman', linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Correlation Between Pre-training and Post-training Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Mean accuracy over epochs
        ax = axes[0, 1]
        ax.plot(stats_df['epoch'], stats_df['mean_pre_acc'], 'o-', label='Pre-training (NSGA-II)', linewidth=2)
        ax.plot(stats_df['epoch'], stats_df['mean_post_acc'], 's-', label='Post-training', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Validation Accuracy (%)')
        ax.set_title('Mean Accuracy Progression')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Delta accuracy over epochs
        ax = axes[1, 0]
        ax.plot(stats_df['epoch'], stats_df['mean_delta_acc'], 'o-', linewidth=2, markersize=8, color='green')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Δ Accuracy (%)')
        ax.set_title('Mean Accuracy Improvement Over Training')
        ax.grid(alpha=0.3)
        
        # Plot 4: Scatter of final epoch
        ax = axes[1, 1]
        final_epoch = correlation_df['epoch'].max()
        final_data = correlation_df[correlation_df['epoch'] == final_epoch]
        
        # Check if we have selection column for coloring
        if 'selection' in final_data.columns:
            colors = {'best_generation': 'red', 'random_generation': 'blue'}
            for selection, color in colors.items():
                subset = final_data[final_data['selection'] == selection]
                if not subset.empty:
                    ax.scatter(subset['pre_val_acc'], subset['val_acc'], 
                              c=color, label=selection.replace('_', ' '), s=100, alpha=0.7, edgecolors='black')
        else:
            # Fallback: color by label if selection not available
            ax.scatter(final_data['pre_val_acc'], final_data['val_acc'], 
                      c='blue', s=100, alpha=0.7, edgecolors='black', label='Candidates')
        
        # Add diagonal line
        min_val = min(final_data['pre_val_acc'].min(), final_data['val_acc'].min())
        max_val = max(final_data['pre_val_acc'].max(), final_data['val_acc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
        
        ax.set_xlabel('Pre-training Accuracy (NSGA-II) (%)')
        ax.set_ylabel(f'Post-training Accuracy (Epoch {final_epoch}) (%)')
        ax.set_title(f'Final Epoch Correlation (Epoch {final_epoch})')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(run_dir / "epoch_correlation_analysis.png", dpi=300)
        plt.close()
        
        print(f"[INFO] Correlation analysis plot saved to {run_dir / 'epoch_correlation_analysis.png'}")
    
    return stats_df.to_dict('records')


def display_final_accuracies(correlation_df: pd.DataFrame, epochs_filter: Optional[List[int]] = None) -> None:
    """Display final training accuracy for each candidate at specified epochs.
    
    Args:
        correlation_df: DataFrame from extract_epoch_correlation
        epochs_filter: Optional list of epochs to display. If None, shows all epochs.
    """
    if correlation_df.empty:
        print("[WARNING] No data to display")
        return
    
    # Get unique epochs, sorted
    available_epochs = sorted(correlation_df['epoch'].unique())
    
    if epochs_filter:
        # Filter to requested epochs that are available
        display_epochs = [e for e in epochs_filter if e in available_epochs]
        if not display_epochs:
            print(f"[WARNING] None of the requested epochs {epochs_filter} are available in the data")
            print(f"[INFO] Available epochs: {available_epochs}")
            return
        missing_epochs = [e for e in epochs_filter if e not in available_epochs]
        if missing_epochs:
            print(f"[WARNING] Requested epochs not found in data: {missing_epochs}")
    else:
        display_epochs = available_epochs
    
    print(f"\n{'='*100}")
    print("FINAL TRAINING ACCURACY BY CANDIDATE AND EPOCH")
    print(f"{'='*100}")
    print(f"Displaying epochs: {display_epochs}")
    print()
    
    # Group by candidate (eval_id)
    candidates = correlation_df['eval_id'].unique()
    
    # Create a summary table
    summary_data = []
    
    for candidate_id in candidates:
        candidate_data = correlation_df[correlation_df['eval_id'] == candidate_id].sort_values('epoch')
        
        if candidate_data.empty:
            continue
        
        # Get candidate metadata from first row
        first_row = candidate_data.iloc[0]
        embed = first_row['embed']
        n_qubits = first_row['n_qubits']
        depth = first_row['depth']
        pre_acc = first_row['pre_val_acc']
        label = first_row['label']
        
        row = {
            'eval_id': candidate_id,
            'label': label,
            'architecture': f"{embed} q{n_qubits} d{depth}",
            'pre_acc': pre_acc,
        }
        
        # Add accuracy for each requested epoch
        for epoch in display_epochs:
            epoch_data = candidate_data[candidate_data['epoch'] == epoch]
            if not epoch_data.empty:
                acc = epoch_data.iloc[0]['val_acc']
                train_acc = epoch_data.iloc[0].get('train_acc', None)
                row[f'epoch_{epoch}_val'] = acc
                if train_acc is not None:
                    row[f'epoch_{epoch}_train'] = train_acc
                row[f'epoch_{epoch}_delta'] = acc - pre_acc
            else:
                row[f'epoch_{epoch}_val'] = None
                row[f'epoch_{epoch}_delta'] = None
        
        summary_data.append(row)
    
    # Create DataFrame and sort by final epoch performance
    summary_df = pd.DataFrame(summary_data)
    final_epoch_col = f'epoch_{display_epochs[-1]}_val'
    if final_epoch_col in summary_df.columns:
        summary_df = summary_df.sort_values(final_epoch_col, ascending=False, na_position='last')
    
    # Display the table
    print(f"\n{'Candidate':<20} {'Label':<25} {'Architecture':<20} {'Pre-Acc':<10}", end='')
    for epoch in display_epochs:
        epoch_str = f"E{epoch:>2} Val"
        print(f"{epoch_str:<12}", end='')
        delta_str = f"Δ{epoch:>2}"
        print(f"{delta_str:<10}", end='')
    print()
    print("-" * (85 + 22 * len(display_epochs)))
    
    for _, row in summary_df.iterrows():
        print(f"{row['eval_id']:<20} {row['label']:<25} {row['architecture']:<20} {row['pre_acc']:>8.2f}%  ", end='')
        for epoch in display_epochs:
            val_col = f'epoch_{epoch}_val'
            delta_col = f'epoch_{epoch}_delta'
            if pd.notna(row.get(val_col)):
                print(f"{row[val_col]:>9.2f}%  ", end='')
                delta = row.get(delta_col, 0)
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                print(f"{delta_str:>8}  ", end='')
            else:
                print(f"{'N/A':>9}   {'N/A':>8}  ", end='')
        print()
    
    # Print summary statistics
    print()
    print(f"{'='*100}")
    print("SUMMARY STATISTICS BY EPOCH")
    print(f"{'='*100}")
    
    stats_data = []
    for epoch in display_epochs:
        epoch_data = correlation_df[correlation_df['epoch'] == epoch]
        if not epoch_data.empty:
            stats_data.append({
                'Epoch': epoch,
                'Mean Val Acc': epoch_data['val_acc'].mean(),
                'Std Val Acc': epoch_data['val_acc'].std(),
                'Min Val Acc': epoch_data['val_acc'].min(),
                'Max Val Acc': epoch_data['val_acc'].max(),
                'Mean Train Acc': epoch_data['train_acc'].mean() if 'train_acc' in epoch_data.columns else None,
                'Mean Δ Acc': epoch_data['delta_acc'].mean(),
                'N Samples': len(epoch_data),
            })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    print()
    
    # Find best candidate at each epoch
    print(f"{'='*100}")
    print("BEST CANDIDATE AT EACH EPOCH")
    print(f"{'='*100}")
    
    for epoch in display_epochs:
        epoch_data = correlation_df[correlation_df['epoch'] == epoch]
        if not epoch_data.empty:
            best_idx = epoch_data['val_acc'].idxmax()
            best = epoch_data.loc[best_idx]
            arch_str = f"{best['embed']} q{best['n_qubits']} d{best['depth']}"
            print(f"Epoch {epoch:>2}: {best['eval_id']:<20} ({best['label']:<25}) "
                  f"Val: {best['val_acc']:>6.2f}% (Δ: {best['delta_acc']:>+6.2f}%) "
                  f"[{arch_str}]")
    print()


# --------------------------
# Main
# --------------------------

def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    
    # If --analyze-run is specified, just analyze that specific run and exit
    if args.analyze_run:
        if args.log_dir is None:
            print(f"[ERROR] --analyze-run requires a log_dir argument")
            sys.exit(1)
        run_dir = args.log_dir / args.analyze_run
        if not run_dir.exists():
            print(f"[ERROR] Run directory not found: {run_dir}")
            sys.exit(1)
        
        # Parse epochs filter if provided
        epochs_filter = None
        if args.analyze_epochs:
            try:
                epochs_filter = [int(e.strip()) for e in args.analyze_epochs.split(',')]
                print(f"[INFO] Analyzing specific epochs: {epochs_filter}")
            except ValueError:
                print(f"[ERROR] Invalid epoch list format: {args.analyze_epochs}")
                print("[ERROR] Expected comma-separated integers, e.g., '1,5,10,20'")
                sys.exit(1)
        
        print(f"[INFO] Analyzing run: {run_dir}")
        try:
            correlation_df = extract_epoch_correlation(run_dir, epochs_filter=epochs_filter)
            if not correlation_df.empty:
                print(f"[INFO] Analysis complete!")
                print(f"[INFO] - Epoch correlation data: {run_dir / 'epoch_correlation.csv'}")
                
                # Show final accuracies if requested
                if args.show_final_accuracy:
                    display_final_accuracies(correlation_df, epochs_filter=epochs_filter)
                
                # Analyze correlation
                analyze_epoch_correlation(correlation_df, run_dir)
                
                # Show summary of final best candidate
                final_best = correlation_df[correlation_df['label'].str.contains('final_best', case=False, na=False)]
                if not final_best.empty:
                    eval_id = final_best.iloc[0]['eval_id']
                    epochs = final_best['epoch'].tolist()
                    accs = final_best['val_acc'].tolist()
                    print(f"\n[INFO] Final best candidate: {eval_id}")
                    print(f"[INFO] Accuracy progression: {' → '.join([f'E{e}: {a:.2f}%' for e, a in zip(epochs, accs)])}")
            else:
                print(f"[WARNING] No data could be extracted from {run_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to analyze run: {e}")
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Determine if we're in random mode
    random_mode = args.random_mode or args.log_dir is None
    
    if random_mode and args.log_dir is not None:
        print(f"[WARNING] Both log_dir and --random-mode specified. Using random mode.")
    
    # Load configuration from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Read all settings from environment
    reuse_folder = _load_env_bool("CORRELATION_REUSE_FOLDER", False)
    no_analysis_folder = _load_env_bool("CORRELATION_NO_ANALYSIS_FOLDER", False)
    save_weights = _load_env_bool("CORRELATION_SAVE_WEIGHTS", False)
    no_plots = _load_env_bool("CORRELATION_NO_PLOTS", False)
    dry_run = _load_env_bool("CORRELATION_DRY_RUN", False)
    
    final_epochs = _load_env_int("CORRELATION_FINAL_EPOCHS", 10)
    final_lr = _load_env_float("CORRELATION_FINAL_LR", None)
    lr_scale = _load_env_float("CORRELATION_LR_SCALE", 1.0) or 1.0
    final_shots = _load_env_int("CORRELATION_FINAL_SHOTS", 1024)
    random_per_gen = _load_env_int("CORRELATION_RANDOM_PER_GEN", 1)
    extra_random = _load_env_int("CORRELATION_EXTRA_RANDOM", 0)
    max_gpus = _load_env_int("CORRELATION_MAX_GPUS", 0)
    workers_per_gpu = _load_env_int("CORRELATION_WORKERS_PER_GPU", 2)
    seed = _load_env_int("CORRELATION_SEED", 1234)
    diverse_seeds = _load_env_bool("CORRELATION_DIVERSE_SEEDS", False)
    train_size = _load_env_int("CORRELATION_TRAIN_SIZE", 0)
    val_size = _load_env_int("CORRELATION_VAL_SIZE", 0)
    
    # Add current directory to sys.path BEFORE resolving module name
    # This ensures Main.py can be found
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    
    hq_module = _resolve_hq_module_name(None)
    
    # CRITICAL: Set IMPORTED_AS_MODULE, RUN_TYPE, and temporary paths BEFORE importing the module
    # This prevents the module from creating directories in the default logs location
    os.environ["IMPORTED_AS_MODULE"] = "true"
    os.environ["RUN_TYPE"] = "correlation"
    # Use /dev/null for CSV paths to prevent file creation during import
    # We'll update these to actual paths after we determine run_dir
    temp_log_dir = Path("/tmp/correlation_temp_import")
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOG_DIR"] = str(temp_log_dir)
    os.environ["DATASET_LOG_DIR"] = str(temp_log_dir)
    os.environ["NSGA_EVAL_CSV"] = str(temp_log_dir / "nsga_evals.csv")
    os.environ["EPOCH_LOG_CSV"] = str(temp_log_dir / "train_epoch_log.csv")
    os.environ["GEN_SUMMARY_CSV"] = str(temp_log_dir / "nsga_gen_summary.csv")
    os.environ["CHECKPOINT_LOG_CSV"] = str(temp_log_dir / "checkpoint_validation.csv")
    os.environ["PROGRESS_LOG"] = str(temp_log_dir / "progress.log")
    
    # Import training module early to get dataset and configuration
    HQMain = importlib.import_module(hq_module)
    
    # Get dataset from environment or Main module
    dataset = os.getenv("DATASET")
    if not dataset:
        dataset = getattr(HQMain, "DATASET", "mnist").lower()
    
    if random_mode:
        print("\n" + "="*70)
        print("RANDOM SAMPLING MODE")
        print("="*70)
        print(f"[INFO] Generating {args.random_samples} random architecture samples")
        print(f"[INFO] Dataset: {dataset}")
        print(f"[INFO] Training module: {hq_module}")
        
        # Create output directory for correlation analysis
        # Format: logs/correlation/{DATASET}/run_{TIMESTAMP}
        dataset_upper = dataset.upper()
        correlation_base = Path("./logs") / "correlation"
        log_dir = correlation_base / dataset_upper
        
        # Create run subdirectory with timestamp
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = log_dir / f"run_{time_stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Copy .env file to run folder for reproducibility
        _copy_env_to_run_folder(run_dir, time_stamp)
        
        print(f"[INFO] Output directory: {log_dir}")
        print(f"[INFO] Run directory: {run_dir}")
        
        # Setup environment for training module (updates LOG_DIR to actual run_dir)
        _ensure_module_environment(dataset, run_dir)
        
        # Reload module to pick up environment changes
        # Clear the module from cache first to force a fresh import
        if hq_module in sys.modules:
            del sys.modules[hq_module]
        # Also clear related modules that might cache paths
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('qnas.') or mod_name == 'qnas':
                del sys.modules[mod_name]
        HQMain = _import_hq_module(hq_module)
        
        # Force update module attributes after reload
        HQMain.LOG_DIR = str(run_dir)
        HQMain.DATASET_LOG_DIR = str(run_dir)
        HQMain.EPOCH_LOG_CSV = str(run_dir / "train_epoch_log.csv")
        HQMain.CHECKPOINT_LOG_CSV = str(run_dir / "checkpoint_validation.csv")
        HQMain.PROGRESS_LOG = str(run_dir / "progress.log")
        
        # Generate random candidates
        cnot_lookup = {name.lower(): idx for idx, name in enumerate(getattr(HQMain, "CNOT_MODES", ["all","odd","even","none"]))}
        candidates = generate_random_candidates(
            n_samples=args.random_samples,
            lr_override=final_lr,
            shots=int(final_shots),
            seed=seed,
            HQMain=HQMain,
        )
        
        evals_df = pd.DataFrame()  # Empty for random mode
        epoch_df = None
        
    else:
        # Original mode: load from NSGA-II logs
        if args.log_dir is None:
            print("[ERROR] log_dir is required when not in random mode")
            print("[INFO] Use --random-mode to generate random samples without NSGA-II logs")
            sys.exit(1)
            
        log_dir = args.log_dir.resolve()
        ensure_required_files(log_dir)
        dataset = infer_dataset_name(log_dir)
        
        # Output structure for correlation analysis
        # Create: logs/correlation/{DATASET}/run_{TIMESTAMP}
        dataset_upper = dataset.upper()
        correlation_base = Path("./logs") / "correlation"
        correlation_log_dir = correlation_base / dataset_upper
        
        if no_analysis_folder:
            run_dir = log_dir
            print(f"[INFO] Skipping run folder creation, using: {run_dir}")
        else:
            if reuse_folder:
                existing_runs = sorted(correlation_log_dir.glob("run_*"))
                if existing_runs:
                    run_dir = existing_runs[-1]
                    print(f"[INFO] Reusing existing correlation run folder: {run_dir}")
                else:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    run_dir = correlation_log_dir / f"run_{stamp}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[INFO] Created new correlation run folder: {run_dir}")
                    # Copy .env file to run folder for reproducibility
                    _copy_env_to_run_folder(run_dir, stamp)
            else:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                run_dir = correlation_log_dir / f"run_{stamp}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Created new correlation analysis run: {run_dir}")
                # Copy .env file to run folder for reproducibility
                _copy_env_to_run_folder(run_dir, stamp)
        
        # Setup environment for training module (updates LOG_DIR to actual run_dir)
        _ensure_module_environment(dataset, run_dir)
        # Reload module to pick up environment changes
        # Clear the module from cache first to force a fresh import
        if hq_module in sys.modules:
            del sys.modules[hq_module]
        # Also clear related modules that might cache paths
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('qnas.') or mod_name == 'qnas':
                del sys.modules[mod_name]
        HQMain = _import_hq_module(hq_module)
        
        # Force update module attributes after reload
        HQMain.LOG_DIR = str(run_dir)
        HQMain.DATASET_LOG_DIR = str(run_dir)
        HQMain.EPOCH_LOG_CSV = str(run_dir / "train_epoch_log.csv")
        HQMain.CHECKPOINT_LOG_CSV = str(run_dir / "checkpoint_validation.csv")
        HQMain.PROGRESS_LOG = str(run_dir / "progress.log")
        
        print(f"[INFO] Analysis dir: {run_dir}")
        print(f"[INFO] Dataset: {dataset or 'default (mnist)'}")
        print(f"[INFO] Training module: {hq_module}")
        
        evals_df, epoch_df = load_nsga_logs(log_dir)
        
        if evals_df.empty:
            print(f"[ERROR] nsga_evals.csv is required for candidate selection in correlation mode")
            print(f"[ERROR] Without it, we cannot select candidates to retrain")
            print(f"[INFO] Options:")
            print(f"[INFO]   1. Provide nsga_evals.csv in {log_dir}")
            print(f"[INFO]   2. Use --random-mode to generate random candidates instead")
            sys.exit(1)
        
        print(f"[INFO] Loaded {len(evals_df)} NSGA evaluations")
        
        cnot_lookup = {name.lower(): idx for idx, name in enumerate(getattr(HQMain, "CNOT_MODES", ["all","odd","even","none"]))}
        
        candidates = select_candidates(
            evals_df,
            random_per_gen=random_per_gen,
            extra_random=extra_random,
            lr_override=final_lr,
            lr_scale=lr_scale,
            shots=int(final_shots),
            cnot_lookup=cnot_lookup,
            seed=seed,
            epoch_df=epoch_df,
        )
    
    # Common validation and settings (works for both modes)
    if final_epochs is None:
        final_epochs = max(10, int(getattr(HQMain, "FINAL_TRAIN_EPOCHS", 3)) * 2)
    if final_shots is None:
        final_shots = int(getattr(HQMain, "FINAL_SHOTS", 0))
    if train_size == 0:
        default_train = int(getattr(HQMain, "FINAL_TRAIN_SUBSET_SIZE", 0))
        train_size = default_train
    if val_size == 0:
        # When val_size is 0, fall back to FINAL_VAL_SUBSET_SIZE (0 means full validation set)
        default_val = int(getattr(HQMain, "FINAL_VAL_SUBSET_SIZE", 0))
        val_size = default_val
    
    if not candidates:
        raise RuntimeError("No candidates were selected; check your logs or filters.")
    
    print(f"\n[INFO] Selected {len(candidates)} candidates")
    
    if not random_mode:
        # Only validate sample diversity for correlation mode
        print(f"[VALIDATION] Ensuring sample diversity:")
        unique_configs = len(set((c.embed, c.n_qubits, c.depth, tuple(c.ent_ranges), 
                                 tuple(c.cnot_modes), round(c.learning_rate, 6)) for c in candidates))
        print(f"[VALIDATION] Unique parameter combinations: {unique_configs}/{len(candidates)}")
        
        if diverse_seeds:
            print(f"[VALIDATION] Diverse training seeds: ENABLED (each job will use different data samples)")
        else:
            print(f"[VALIDATION] Diverse training seeds: DISABLED (all jobs use same data samples)")
            print(f"[VALIDATION] To enable diverse data samples, set CORRELATION_DIVERSE_SEEDS=true in .env")
        
        # Check generation spread
        gen_spread = len(set(c.generation for c in candidates))
        print(f"[VALIDATION] Generation spread: {gen_spread} different generations")
        
        # Check parameter variety
        embed_variety = len(set(c.embed for c in candidates))
        qubit_variety = len(set(c.n_qubits for c in candidates))
        depth_variety = len(set(c.depth for c in candidates))
        print(f"[VALIDATION] Parameter variety: {embed_variety} embeds, {qubit_variety} qubit counts, {depth_variety} depths")

    selected_df = pd.DataFrame([{
        "eval_id": c.eval_id,
        "generation": c.generation,
        "selection": c.selection,
        "embed": c.embed,
        "n_qubits": c.n_qubits,
        "depth": c.depth,
        "ent_ranges": "-".join(map(str, c.ent_ranges)),
        "cnot_modes": "-".join(getattr(HQMain, "CNOT_MODES", ["all","odd","even","none"])[m] for m in c.cnot_modes),
        "original_learning_rate": c.original_learning_rate,
        "final_learning_rate": c.learning_rate,
        "shots": c.shots,
        "pre_val_acc": c.pre_val_acc,
        "pre_val_loss": c.pre_val_loss,
        "f2_circuit_cost": c.meta.get("f2_circuit_cost"),
        "f3_n_subcircuits": c.meta.get("f3_n_subcircuits"),
        "seconds": c.meta.get("seconds"),
    } for c in candidates])
    selected_df.to_csv(run_dir / "selected_candidates.csv", index=False)

    results: List[RetrainResult] = []
    if not dry_run:
        # Get checkpoint validation settings
        checkpoint_enabled = os.getenv("CHECKPOINT_CORRELATION_ENABLED", "true").lower() in ("true", "1", "yes")
        checkpoint_train_sizes = os.getenv("CHECKPOINT_TRAIN_SIZES", "512,1024,2048,4096,8196,16392,32768,full")
        checkpoint_target_epochs = os.getenv("CHECKPOINT_TARGET_EPOCHS", "1,3,5,10")
        
        print(f"[INFO] Retraining: epochs={final_epochs}, shots={final_shots}, max_gpus={max_gpus or 'ALL'}, workers/GPU={workers_per_gpu}")
        print(f"[INFO] Checkpoint validation: {'ENABLED' if checkpoint_enabled else 'DISABLED'}")
        if checkpoint_enabled:
            print(f"[INFO]   - Train sizes: {checkpoint_train_sizes}")
            print(f"[INFO]   - Target epochs: {checkpoint_target_epochs}")
            print(f"[INFO]   - Output: {run_dir / 'checkpoint_validation.csv'}")
        
        # Create a simple namespace to pass parameters
        class Args:
            pass
        retrain_args = Args()
        retrain_args.final_epochs = final_epochs
        retrain_args.final_lr = final_lr
        retrain_args.final_shots = final_shots
        retrain_args.max_gpus = max_gpus
        retrain_args.workers_per_gpu = workers_per_gpu
        retrain_args.save_weights = save_weights
        retrain_args.diverse_seeds = diverse_seeds
        retrain_args.train_size = train_size
        retrain_args.val_size = val_size
        retrain_args.hq_module = hq_module
        results = retrain_candidates(candidates, retrain_args, run_dir, dataset)
    else:
        print("[INFO] Dry-run mode: skipping retraining (set CORRELATION_DRY_RUN=false in .env to enable)")
        # Create empty results for dry-run
        results = []

    # Merge & save
    if results:
        res_df = pd.DataFrame([{
            "eval_id": r.eval_id,
            "post_val_acc": r.post_val_acc,
            "post_val_loss": r.post_val_loss,
            "epochs": r.epochs,
            "lr_used": r.lr_used,
            "shots_used": r.shots_used,
            "device": r.device,
            "retrain_seconds": r.seconds,
            "weights_path": r.weights_path,
            "error": r.error,
        } for r in results])
    else:
        res_df = pd.DataFrame(columns=["eval_id","post_val_acc","post_val_loss","epochs","lr_used","shots_used","device","retrain_seconds","weights_path","error"])

    combined = selected_df.merge(res_df, on="eval_id", how="left")
    combined["delta_val_acc"]  = combined["post_val_acc"]  - combined["pre_val_acc"]
    combined["delta_val_loss"] = combined["post_val_loss"] - combined["pre_val_loss"]
    combined.to_csv(run_dir / "correlation_results.csv", index=False)

    # Simple correlation metrics
    def _corr(df: pd.DataFrame) -> Dict[str, float]:
        if len(df) < 2: return {"pearson": float("nan"), "spearman": float("nan"), "order_consistency": float("nan")}
        pre = df["pre_val_acc"].to_numpy(); post = df["post_val_acc"].to_numpy()
        pearson = float(np.corrcoef(pre, post)[0, 1])
        pre_rank = pd.Series(pre).rank(method="average").to_numpy()
        post_rank = pd.Series(post).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(pre_rank, post_rank)[0, 1])
        consistent = 0; comparable = 0
        n = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                dp = pre[i] - pre[j]; dq = post[i] - post[j]
                if dp == 0 or dq == 0: continue
                comparable += 1
                if dp * dq > 0: consistent += 1
        return {"pearson": pearson, "spearman": spearman, "order_consistency": (consistent / comparable) if comparable else float("nan")}

    stats = _corr(combined.dropna(subset=["post_val_acc"]))
    
    # Determine generation count (different for random vs correlation mode)
    if random_mode:
        generation_count = 0  # No generations in random mode
    else:
        generation_count = int(
            evals_df["generation"].replace(-1, np.nan).max() + 1
            if not evals_df["generation"].replace(-1, np.nan).isna().all()
            else len(evals_df)
        )
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "log_dir": str(log_dir),
        "dataset": dataset,
        "module": hq_module,
        "mode": "random" if random_mode else "correlation",
        "config": {
            "final_epochs": final_epochs,
            "final_lr": final_lr,
            "lr_scale": lr_scale,
            "final_shots": final_shots,
            "random_per_gen": random_per_gen if not random_mode else 0,
            "extra_random": extra_random if not random_mode else 0,
            "random_samples": args.random_samples if random_mode else 0,
            "max_gpus": max_gpus,
            "workers_per_gpu": workers_per_gpu,
            "seed": seed,
            "train_size": train_size,
            "val_size": val_size,
            "dry_run": dry_run,
        },
        "counts": {
            "total_evals": len(evals_df) if not random_mode else 0,
            "generations": generation_count,
            "selected": len(combined),
            "retrained": int(combined["post_val_acc"].notna().sum()),
            "failures": int(combined["error"].notna().sum()) if "error" in combined else 0,
        },
        "metrics": {
            "pearson": stats["pearson"],
            "spearman": stats["spearman"],
            "order_consistency": stats["order_consistency"],
            "mean_delta_acc": float(combined["delta_val_acc"].dropna().mean()) if combined["delta_val_acc"].notna().any() else float("nan"),
            "median_delta_acc": float(combined["delta_val_acc"].dropna().median()) if combined["delta_val_acc"].notna().any() else float("nan"),
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not no_plots and not combined.empty and combined["post_val_acc"].notna().any():
        plot_scatter(combined.dropna(subset=["post_val_acc"]), run_dir / "scatter_before_after.png")
        plot_trend  (combined.dropna(subset=["post_val_acc"]), run_dir / "trend_best_per_generation.png")

    print("[INFO] Analysis complete. Summary →", run_dir / "summary.json")
    
    # Print checkpoint validation summary if enabled
    checkpoint_enabled = os.getenv("CHECKPOINT_CORRELATION_ENABLED", "true").lower() in ("true", "1", "yes")
    if checkpoint_enabled and (run_dir / "checkpoint_validation.csv").exists():
        checkpoint_df = pd.read_csv(run_dir / "checkpoint_validation.csv")
        if not checkpoint_df.empty:
            print("\n[INFO] Checkpoint Validation Summary:")
            print(f"[INFO] - Total checkpoints: {len(checkpoint_df)}")
            print(f"[INFO] - Candidates evaluated: {checkpoint_df['eval_id'].nunique()}")
            print(f"[INFO] - Epochs tracked: {sorted(checkpoint_df['epoch'].unique())}")
            print(f"[INFO] - Train sizes: {sorted(checkpoint_df['checkpoint_train_size'].unique())}")
            print(f"[INFO] - Checkpoint data: {run_dir / 'checkpoint_validation.csv'}")
    
    # Generate epoch-by-epoch correlation CSV (only in correlation mode, not random mode)
    if not random_mode:
        print("\n[INFO] Generating epoch-by-epoch correlation data...")
        try:
            correlation_df = extract_epoch_correlation(run_dir)
            if not correlation_df.empty:
                print("[INFO] Epoch correlation CSV generated!")
                print(f"[INFO] - Epoch correlation data: {run_dir / 'epoch_correlation.csv'}")
                
                # Show summary of final best candidate
                final_best = correlation_df[correlation_df['label'] == 'final_best']
                if not final_best.empty:
                    eval_id = final_best.iloc[0]['eval_id']
                    epochs = final_best['epoch'].tolist()
                    accs = final_best['val_acc'].tolist()
                    f2 = final_best.iloc[0]['f2_circuit_cost']
                    f3 = final_best.iloc[0]['f3_n_subcircuits']
                    print(f"[INFO] Final best candidate: {eval_id} (f2={f2}, f3={f3})")
                    print(f"[INFO] Accuracy progression: {' → '.join([f'E{e}: {a:.2f}%' for e, a in zip(epochs, accs)])}")
            else:
                print("[WARNING] Could not generate epoch correlation data (no data available)")
        except Exception as e:
            print(f"[WARNING] Failed to generate epoch correlation data: {e}")
    else:
        print("\n[INFO] Skipping epoch correlation analysis (random mode has no pre-training baseline)")
        print(f"[INFO] Results saved to: {run_dir}")
        print(f"[INFO] - Training logs: {run_dir / 'train_epoch_log.csv'}")
        print(f"[INFO] - Checkpoint validation: {run_dir / 'checkpoint_validation.csv'}")
        print(f"[INFO] - Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
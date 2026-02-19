#!/usr/bin/env python3
"""Run final training on Pareto-optimal configurations from an existing NSGA run."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path (go up 3 levels from scripts/training/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _setup_run_environment(csv_path: Path) -> Path:
    """Configure environment so logging reuses an existing run directory."""
    run_dir = csv_path.parent.resolve()

    # logs root: logs/nsga-ii/{DATASET}/run_xxx -> parents[2] is logs/
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

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final training on Pareto-optimal NSGA-II configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/training/run_final_training.py logs/nsga-ii/MNIST/run_20260101-120000/nsga_evals.csv
  python scripts/training/run_final_training.py logs/nsga-ii/MNIST/run_20260101-120000/nsga_evals.csv --gpus 0 1
        """,
    )
    parser.add_argument("csv_path", type=Path, help="Path to nsga_evals.csv file")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (default: all available GPUs)")
    parser.add_argument("--objectives", type=str, nargs="+", default=None,
                        help="Objectives for Pareto front (default from config)")

    args = parser.parse_args()
    csv_path = args.csv_path.resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    run_dir = _setup_run_environment(csv_path)

    from qnas.models.config import QConfig
    from qnas.nsga2.runner import final_train
    from qnas.utils import config as cfg
    from qnas.utils import logging_utils as log_utils

    cfg.set_dataset_log_dir(str(run_dir), create=False)
    log_utils.refresh_logging_paths(str(run_dir))

    fallback_cfg = QConfig(
        embed_kind="angle-x",
        n_qubits=2,
        depth=1,
        ent_ranges=[1],
        cnot_modes=[0],
        learning_rate=1e-3,
        shots=cfg.FINAL_SHOTS,
    )

    print(f"Using existing run directory: {run_dir}")
    final_train(fallback_cfg, csv_path=csv_path, gpus=args.gpus, objectives=args.objectives)


if __name__ == "__main__":
    main()

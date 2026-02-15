#!/usr/bin/env python3
"""
QNAS Analysis and Visualization Suite

Provides comprehensive visualization and analysis for NSGA-II optimization
results of Hybrid Quantum Neural Networks.

Usage:
    python scripts/analysis/plot.py                    # Generate all plots
    python scripts/analysis/plot.py --folder=PATH      # Specific log folder
    python scripts/analysis/plot.py --font-size=14     # Custom font size
    python scripts/analysis/plot.py --svg              # SVG output format
    python scripts/analysis/plot.py list               # List available runs

Output:
    All plots are saved to outputs/figures/ with consistent naming.
"""

import os
# Prevent src modules from creating log directories when imported
os.environ["IMPORTED_AS_MODULE"] = "true"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
import subprocess
import sys
import shutil
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from PIL import Image
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

# Import config values
try:
    _plot_file = os.path.abspath(__file__)
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(_plot_file)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from qnas.utils.config import CUT_TARGET_QUBITS
except ImportError:
    CUT_TARGET_QUBITS = 5

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PlotConfig:
    """Centralized configuration for all plots."""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))  # Will be set to run-specific dir
    font_size: int = 14
    file_ext: str = "png"
    dpi: int = 300
    figsize_small: tuple = (8, 6)
    figsize_medium: tuple = (12, 8)
    figsize_large: tuple = (16, 10)
    figsize_wide: tuple = (18, 6)
    colors: List = field(default_factory=lambda: sns.color_palette("colorblind"))
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        # Don't create directory here - it will be created when actually saving files
        # This prevents creating "outputs/figures" before we can set the run-specific path
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist. Call this after setting the final path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
PLOT_CONFIG = PlotConfig()


def save_figure(fig, name: str, config: PlotConfig = None, category: str = None):
    """Save figure with consistent naming and logging.
    
    Args:
        fig: Matplotlib figure
        name: Base name for the file (e.g., 'evolution_accuracy')
        config: PlotConfig instance (uses global if None)
        category: Optional category prefix (e.g., 'pareto', 'evolution')
    """
    if config is None:
        config = PLOT_CONFIG
    
    # Build filename: {category}_{name}.{ext} or just {name}.{ext}
    if category:
        filename = f"{category}_{name}.{config.file_ext}"
    else:
        filename = f"{name}.{config.file_ext}"
    
    filepath = config.output_dir / filename
    
    if config.file_ext == "svg":
        fig.savefig(filepath, format='svg', bbox_inches='tight')
    else:
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
    
    plt.close(fig)
    print(f"  [SAVED] {filename}")
    return filepath


def log_section(title: str):
    """Print section header for consistent logging."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def log_step(step_num: int, description: str):
    """Print step info for consistent logging."""
    print(f"\n[{step_num}] {description}")


# Legacy alias for compatibility
FONT_SIZE = 14


class NSGAPlotter:
    """Class to handle all plotting operations for NSGA-II evaluation data."""
    
    def __init__(self, log_folder=None, csv_path="logs/nsga_evals.csv", dataset=None, run_dir=None):
        """Initialize with data loading.
        
        Parameters:
        -----------
        log_folder : str, optional
            Specific folder name within logs directory (e.g., 'nsga-ii', 'correlation')
        csv_path : str, default="logs/nsga_evals.csv"
            Default path for evaluation data (used when log_folder is None)
        dataset : str, optional
            Dataset name (e.g., 'MNIST', 'SVHN') - required for nested structures
        run_dir : str, optional
            Run directory name (e.g., 'run_20251212-140530') - if None, uses most recent
        """
        if log_folder:
            self.log_folder = log_folder
            # Check if log_folder contains slashes (is a path, not just a folder name)
            if '/' in log_folder or '\\' in log_folder:
                # Path provided - check if it needs logs/ prefix
                log_folder_path = Path(log_folder)
                
                # If it's a relative path and doesn't start with 'logs/', prepend it
                if not log_folder_path.is_absolute():
                    # Try with logs/ prefix first
                    logs_path = Path('logs') / log_folder
                    if logs_path.exists():
                        log_folder_path = logs_path
                    else:
                        # Otherwise resolve as-is
                        log_folder_path = log_folder_path.resolve()
                
                # Check if this path exists and contains nsga_evals.csv
                csv_file = log_folder_path / "nsga_evals.csv"
                if csv_file.exists():
                    # It's a run directory
                    self.csv_path = csv_file
                    self.gen_summary_path = log_folder_path / "nsga_gen_summary.csv"
                    self.run_dir = log_folder_path
                    self._paths_set = True  # Mark that paths are set
                    print(f"  Log folder: {log_folder}")
                    print(f"  Run directory: {log_folder_path.name}")
                    print(f"  CSV path: {self.csv_path}")
                elif log_folder_path.exists() and log_folder_path.is_dir():
                    # Look for run directories or CSV files inside
                    if log_folder_path.name.startswith('run_'):
                        # It's a run directory but CSV might be missing
                        self.csv_path = log_folder_path / "nsga_evals.csv"
                        self.gen_summary_path = log_folder_path / "nsga_gen_summary.csv"
                        self.run_dir = log_folder_path
                    else:
                        # Look for run directories inside
                        runs = sorted([d for d in log_folder_path.iterdir() if d.is_dir() and d.name.startswith('run_')], 
                                     key=lambda x: x.stat().st_mtime, reverse=True)
                        if runs:
                            run_path = runs[0]
                            self.csv_path = run_path / "nsga_evals.csv"
                            self.gen_summary_path = run_path / "nsga_gen_summary.csv"
                            self.run_dir = run_path
                            self._paths_set = True
                            print(f"  Log folder: {log_folder}")
                            print(f"  Run directory: {run_path.name}")
                        else:
                            # Try as-is
                            self.csv_path = log_folder_path / "nsga_evals.csv"
                            self.gen_summary_path = log_folder_path / "nsga_gen_summary.csv"
                            self.run_dir = log_folder_path
                            self._paths_set = True
                else:
                    # Path doesn't exist - try as relative path from current directory
                    self.csv_path = log_folder_path / "nsga_evals.csv"
                    self.gen_summary_path = log_folder_path / "nsga_gen_summary.csv"
                    self.run_dir = log_folder_path
                    self._paths_set = True
                
                # If we successfully set paths above, skip the rest
                if hasattr(self, '_paths_set') and self._paths_set:
                    pass  # Paths already set, skip nested/flat structure handling
            # Handle nested structure: logs/nsga-ii/{DATASET}/run_{TIMESTAMP}/
            if not (hasattr(self, '_paths_set') and self._paths_set) and log_folder in ['nsga-ii', 'correlation']:
                # Need to find dataset and run directory
                base_path = Path(f"logs/{log_folder}")
                if dataset:
                    dataset_path = base_path / dataset
                else:
                    # Find all datasets
                    datasets = [d.name for d in base_path.iterdir() if d.is_dir()]
                    if not datasets:
                        print(f"[ERROR] No datasets found in {base_path}")
                        self.csv_path = Path(f"logs/{log_folder}/nsga_evals.csv")
                        self.gen_summary_path = Path(f"logs/{log_folder}/nsga_gen_summary.csv")
                        self.data = None
                        return
                    elif len(datasets) == 1:
                        dataset = datasets[0]
                        print(f"Using dataset: {dataset}")
                    else:
                        print(f"Multiple datasets found:")
                        for i, ds in enumerate(datasets, 1):
                            print(f"   {i}. {ds}")
                        try:
                            choice = input(f"Select dataset (1-{len(datasets)}) or press Enter for {datasets[0]}: ").strip()
                            if choice.isdigit() and 1 <= int(choice) <= len(datasets):
                                dataset = datasets[int(choice) - 1]
                            else:
                                dataset = datasets[0]
                            print(f"[OK] Selected dataset: {dataset}")
                        except (EOFError, KeyboardInterrupt):
                            # Non-interactive mode - use first dataset
                            dataset = datasets[0]
                            print(f"[OK] Auto-selected dataset: {dataset} (non-interactive mode)")
                
                dataset_path = base_path / dataset
                if run_dir:
                    run_path = dataset_path / run_dir
                else:
                    # Find most recent run directory
                    runs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('run_')], 
                                 key=lambda x: x.stat().st_mtime, reverse=True)
                    if not runs:
                        print(f"[ERROR] No run directories found in {dataset_path}")
                        self.csv_path = Path(f"logs/{log_folder}/nsga_evals.csv")
                        self.gen_summary_path = Path(f"logs/{log_folder}/nsga_gen_summary.csv")
                        self.data = None
                        return
                    run_path = runs[0]
                    print(f"Using most recent run: {run_path.name}")
                
                self.csv_path = run_path / "nsga_evals.csv"
                self.gen_summary_path = run_path / "nsga_gen_summary.csv"
                self.run_dir = run_path
                self._paths_set = True
            elif not (hasattr(self, '_paths_set') and self._paths_set):
                # Old flat structure (only if paths not already set)
                self.csv_path = Path(f"logs/{log_folder}/nsga_evals.csv")
                self.gen_summary_path = Path(f"logs/{log_folder}/nsga_gen_summary.csv")
                self.run_dir = Path(f"logs/{log_folder}")
                self._paths_set = True
            
            # Print summary (only if not already printed above)
            if not (hasattr(self, '_paths_printed') and self._paths_printed):
                print(f"Using log folder: {log_folder}")
                if dataset:
                    print(f"Dataset: {dataset}")
                if hasattr(self, 'run_dir') and self.run_dir:
                    print(f"Run directory: {self.run_dir.name}")
                self._paths_printed = True
        else:
            self.log_folder = None
            self.csv_path = Path(csv_path)
            self.gen_summary_path = Path("logs/nsga_gen_summary.csv")
            self.run_dir = None
        
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the evaluation data."""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.data)} evaluations from {self.csv_path}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Add derived columns for better analysis
            self.data['generation'] = self.data['gen_est']
            self.data['accuracy_percent'] = self.data['val_acc']
            self.data['error_rate'] = 100 - self.data['val_acc']
            self.data['efficiency'] = self.data['val_acc'] / self.data['seconds']  # acc per second
            self.data['complexity_score'] = self.data['n_qubits'] * self.data['depth']
            
        except FileNotFoundError:
            print(f"Error: Could not find {self.csv_path}")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    def create_output_dir(self, output_dir="outputs/figures"):
        """Create output directory for plots."""
        # Always use outputs/figures for consolidated output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return str(output_path)
    
    @staticmethod
    def list_log_folders():
        """List all available log folders."""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("[ERROR] logs directory not found")
            return []
        
        folders = [f.name for f in logs_dir.iterdir() if f.is_dir()]
        return sorted(folders)
    
    def plot_evolution_progress(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot how key metrics evolve across generations.
        
        Parameters:
        -----------
        output_dir : str, default="plots"
            Directory to save the plot
        font_size : int, default=12
            Base font size for all text elements
        """
        # Create the main plot with consistent seaborn styling
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Group by generation and calculate statistics
            gen_stats = self.data.groupby('generation').agg({
                'val_acc': ['mean', 'max', 'std'],
                'val_loss': ['mean', 'min', 'std'],
                'seconds': ['mean'],
                'f2_circuit_cost': ['mean']
            }).round(3)
            
            generations = gen_stats.index
            
            # Accuracy evolution
            axes[0,0].plot(generations, gen_stats[('val_acc', 'mean')], 'o-', label='Mean Accuracy', linewidth=2)
            axes[0,0].fill_between(generations, 
                                  gen_stats[('val_acc', 'mean')] - gen_stats[('val_acc', 'std')],
                                  gen_stats[('val_acc', 'mean')] + gen_stats[('val_acc', 'std')],
                                  alpha=0.3)
            axes[0,0].plot(generations, gen_stats[('val_acc', 'max')], 's-', label='Best Accuracy', linewidth=2)
            axes[0,0].set_xlabel('Generation', fontsize=font_size)
            axes[0,0].set_ylabel('Validation Accuracy (%)', fontsize=font_size)
            axes[0,0].legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            axes[0,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,0].grid(True, alpha=0.3)
            
            # Loss evolution
            axes[0,1].plot(generations, gen_stats[('val_loss', 'mean')], 'o-', label='Mean Loss', linewidth=2)
            axes[0,1].fill_between(generations,
                                  gen_stats[('val_loss', 'mean')] - gen_stats[('val_loss', 'std')],
                                  gen_stats[('val_loss', 'mean')] + gen_stats[('val_loss', 'std')],
                                  alpha=0.3)
            axes[0,1].plot(generations, gen_stats[('val_loss', 'min')], 's-', label='Best Loss', linewidth=2)
            axes[0,1].set_xlabel('Generation', fontsize=font_size)
            axes[0,1].set_ylabel('Validation Loss', fontsize=font_size)
            axes[0,1].legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            axes[0,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,1].grid(True, alpha=0.3)
            
            # Training time evolution
            axes[1,0].plot(generations, gen_stats[('seconds', 'mean')], 'o-', linewidth=2, color='green')
            axes[1,0].set_xlabel('Generation', fontsize=font_size)
            axes[1,0].set_ylabel('Average Training Time (s)', fontsize=font_size)
            axes[1,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,0].grid(True, alpha=0.3)
            
            # Circuit complexity evolution
            axes[1,1].plot(generations, gen_stats[('f2_circuit_cost', 'mean')], 'o-', linewidth=2, color='red')
            axes[1,1].set_xlabel('Generation', fontsize=font_size)
            axes[1,1].set_ylabel('Average Circuit Cost', fontsize=font_size)
            axes[1,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"{output_dir}/evolution_progress.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def plot_generation_summary(self, output_dir="outputs/figures", gen_summary_path=None, font_size=12, file_ext="png"):
        """Plot evolution progress using generation summary data with seaborn styling.
        
        Parameters:
        -----------
        output_dir : str, default="plots"
            Directory to save the plot
        gen_summary_path : str, optional
            Path to the generation summary CSV file. If None, uses the instance's gen_summary_path
        font_size : int, default=12
            Base font size for all text elements (title will be +2, labels will be base size)
        """
        if gen_summary_path is None:
            gen_summary_path = self.gen_summary_path
        
        try:
            # Read the generation summary data
            df = pd.read_csv(gen_summary_path)
            print(f"Loaded generation summary: {len(df)} generations from {gen_summary_path}")
            
            # Convert 1-accuracy to accuracy for better interpretation
            df['best_accuracy'] = 1 - df['best_1_minus_acc']
            df['median_accuracy'] = 1 - df['median_1_minus_acc']
            
            # Create the plot with seaborn styling
            with sns.axes_style("whitegrid"):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                
                # Plot 1: Accuracy evolution
                sns.lineplot(data=df, x='generation', y='best_accuracy', 
                           marker='o', linewidth=2, markersize=8, 
                           color='#d62728', label='Best-of-generation', ax=ax1)
                sns.lineplot(data=df, x='generation', y='median_accuracy', 
                           marker='s', linewidth=2, markersize=8, 
                           color='#1f77b4', label='Median validation accuracy', ax=ax1)
                
                ax1.set_ylabel('Validation Accuracy', fontsize=font_size, fontweight='bold')

                ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=font_size-1)
                ax1.grid(True, alpha=0.3)
                
                # Customize tick labels
                ax1.tick_params(axis='both', which='major', labelsize=font_size-1)
                
                # Set reasonable y-limits based on data
                y_min = min(df['median_accuracy'].min(), df['best_accuracy'].min()) - 0.02
                y_max = max(df['median_accuracy'].max(), df['best_accuracy'].max()) + 0.02
                ax1.set_ylim(y_min, y_max)
                
                # Plot 2: Circuit cost evolution
                sns.lineplot(data=df, x='generation', y='best_cost', 
                           marker='o', linewidth=2, markersize=8, 
                           color='#2ca02c', label='Best circuit cost (F2)', ax=ax2)
                
                ax2.set_xlabel('Generation', fontsize=font_size, fontweight='bold')
                ax2.set_ylabel('Circuit Cost', fontsize=font_size, fontweight='bold')
                ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=font_size-1)
                ax2.grid(True, alpha=0.3)
                
                # Customize tick labels
                ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
                
                # Set integer ticks for generations
                ax2.set_xticks(range(1, int(df['generation'].max()) + 1))
                
                # Improve overall appearance
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.3)
                
                # Save the plot
                filename = f"{output_dir}/nsga_evolution_progress.{file_ext}"
                if file_ext == "svg":
                    plt.savefig(filename, format='svg', bbox_inches='tight')
                else:
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"[OK] Generation summary plot saved to {filename}")
                
        except FileNotFoundError:
            print(f"[ERROR] Could not find generation summary file: {gen_summary_path}")
        except Exception as e:
            print(f"[ERROR] Error creating generation summary plot: {e}")
    
    def plot_generation_summary_font_variants(self, output_dir="outputs/figures", small_font=10, medium_font=12, large_font=16):
        """Generate generation summary plots with different font sizes for comparison.
        
        Parameters:
        -----------
        output_dir : str, default="plots"
            Directory to save the plots
        small_font : int, default=10
            Small font size
        medium_font : int, default=12
            Medium font size (default)
        large_font : int, default=16
            Large font size
        """
        print("Creating generation summary plots with different font sizes...")
        
        # Create plots with different font sizes
        sizes = [
            (small_font, "small"),
            (medium_font, "medium"), 
            (large_font, "large")
        ]
        
        for font_size, size_name in sizes:
            print(f"  📝 Creating {size_name} font version (size {font_size})...")
            
            # Create a modified filename
            modified_path = f"{output_dir}/nsga_evolution_progress_{size_name}_font.png"
            
            # Plot with specific font size
            self.plot_generation_summary(output_dir, font_size=font_size)
            
            # Rename the file to include font size info
            import os
            default_path = f"{output_dir}/nsga_evolution_progress.png"
            if os.path.exists(default_path):
                os.rename(default_path, modified_path)
                print(f"    [OK] Saved as: {modified_path}")
        
        print(f"Generated 3 versions with different font sizes in {output_dir}/")
    
    def plot_pareto_front(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot Pareto front analysis for multi-objective optimization."""
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy vs Circuit Cost
            scatter = axes[0,0].scatter(self.data['f2_circuit_cost'], self.data['val_acc'], 
                                       c=self.data['generation'], cmap='viridis', alpha=0.7)
            axes[0,0].set_xlabel('Circuit Cost (f2)', fontsize=font_size, fontweight='bold')
            axes[0,0].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,0].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[0,0])
            cbar.set_label('Generation', fontsize=font_size-1)
            cbar.ax.tick_params(labelsize=font_size-2)
            
            # Accuracy vs Training Time
            scatter = axes[0,1].scatter(self.data['seconds'], self.data['val_acc'],
                                       c=self.data['generation'], cmap='viridis', alpha=0.7)
            axes[0,1].set_xlabel('Training Time (seconds)', fontsize=font_size, fontweight='bold')
            axes[0,1].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,1].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[0,1])
            cbar.set_label('Generation', fontsize=font_size-1)
            cbar.ax.tick_params(labelsize=font_size-2)
            
            # 3D objective space (using f1, f2, f3)
            scatter = axes[1,0].scatter(self.data['f1_1_minus_acc'], self.data['f2_circuit_cost'],
                                       c=self.data['f3_n_subcircuits'], cmap='plasma', alpha=0.7)
            axes[1,0].set_xlabel('F1: 1-Accuracy', fontsize=font_size, fontweight='bold')
            axes[1,0].set_ylabel('F2: Circuit Cost', fontsize=font_size, fontweight='bold')
            axes[1,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,0].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[1,0])
            cbar.set_label('F3: Subcircuits', fontsize=font_size-1)
            cbar.ax.tick_params(labelsize=font_size-2)
            
            # Efficiency analysis
            scatter = axes[1,1].scatter(self.data['complexity_score'], self.data['efficiency'],
                                     c=self.data['generation'], cmap='viridis', alpha=0.7)
            axes[1,1].set_xlabel('Complexity Score (qubits × depth)', fontsize=font_size, fontweight='bold')
            axes[1,1].set_ylabel('Efficiency (acc/second)', fontsize=font_size, fontweight='bold')
            axes[1,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"{output_dir}/pareto_analysis.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def plot_hyperparameter_analysis(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Analyze the impact of different hyperparameters."""
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Embedding type analysis
            embed_stats = self.data.groupby('embed')['val_acc'].agg(['mean', 'std', 'count'])
            axes[0,0].bar(embed_stats.index, embed_stats['mean'], yerr=embed_stats['std'], capsize=5)
            axes[0,0].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,0].tick_params(axis='x', rotation=45, labelsize=font_size-1)
            axes[0,0].tick_params(axis='y', labelsize=font_size-1)
            axes[0,0].grid(True, alpha=0.3)
            
            # Number of qubits analysis
            qubit_stats = self.data.groupby('n_qubits')['val_acc'].agg(['mean', 'std'])
            axes[0,1].errorbar(qubit_stats.index, qubit_stats['mean'], yerr=qubit_stats['std'], 
                              marker='o', capsize=5, linewidth=2)
            axes[0,1].set_xlabel('Number of Qubits', fontsize=font_size, fontweight='bold')
            axes[0,1].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,1].grid(True, alpha=0.3)
            
            # Circuit depth analysis
            depth_stats = self.data.groupby('depth')['val_acc'].agg(['mean', 'std'])
            axes[0,2].errorbar(depth_stats.index, depth_stats['mean'], yerr=depth_stats['std'],
                              marker='s', capsize=5, linewidth=2)
            axes[0,2].set_xlabel('Circuit Depth', fontsize=font_size, fontweight='bold')
            axes[0,2].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,2].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,2].grid(True, alpha=0.3)
            
            # Learning rate impact
            lr_bins = pd.cut(self.data['learning_rate'], bins=10)
            lr_stats = self.data.groupby(lr_bins)['val_acc'].mean()
            axes[1,0].plot(range(len(lr_stats)), lr_stats.values, 'o-', linewidth=2)
            axes[1,0].set_xlabel('Learning Rate Bins', fontsize=font_size, fontweight='bold')
            axes[1,0].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[1,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,0].grid(True, alpha=0.3)
            
            # Training time vs qubits
            sns.boxplot(data=self.data, x='n_qubits', y='seconds', ax=axes[1,1])
            axes[1,1].set_xlabel('Number of Qubits', fontsize=font_size, fontweight='bold')
            axes[1,1].set_ylabel('Training Time (seconds)', fontsize=font_size, fontweight='bold')
            axes[1,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,1].grid(True, alpha=0.3)
            
            # Circuit cost vs complexity
            axes[1,2].scatter(self.data['complexity_score'], self.data['f2_circuit_cost'], alpha=0.6)
            axes[1,2].set_xlabel('Complexity Score (qubits × depth)', fontsize=font_size, fontweight='bold')
            axes[1,2].set_ylabel('Circuit Cost', fontsize=font_size, fontweight='bold')
            axes[1,2].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"{output_dir}/hyperparameter_analysis.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def plot_performance_distributions(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot distributions of key performance metrics."""
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy distribution
            axes[0,0].hist(self.data['val_acc'], bins=20, alpha=0.7, edgecolor='black')
            axes[0,0].axvline(self.data['val_acc'].mean(), color='red', linestyle='--', 
                             label=f'Mean: {self.data["val_acc"].mean():.2f}%')
            axes[0,0].set_xlabel('Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,0].set_ylabel('Frequency', fontsize=font_size, fontweight='bold')
            axes[0,0].legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            axes[0,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,0].grid(True, alpha=0.3)
            
            # Training time distribution
            axes[0,1].hist(self.data['seconds'], bins=20, alpha=0.7, edgecolor='black', color='green')
            axes[0,1].axvline(self.data['seconds'].mean(), color='red', linestyle='--',
                             label=f'Mean: {self.data["seconds"].mean():.2f}s')
            axes[0,1].set_xlabel('Time (seconds)', fontsize=font_size, fontweight='bold')
            axes[0,1].set_ylabel('Frequency', fontsize=font_size, fontweight='bold')
            axes[0,1].legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            axes[0,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,1].grid(True, alpha=0.3)
            
            # Loss distribution
            axes[1,0].hist(self.data['val_loss'], bins=20, alpha=0.7, edgecolor='black', color='orange')
            axes[1,0].axvline(self.data['val_loss'].mean(), color='red', linestyle='--',
                             label=f'Mean: {self.data["val_loss"].mean():.3f}')
            axes[1,0].set_xlabel('Loss', fontsize=font_size, fontweight='bold')
            axes[1,0].set_ylabel('Frequency', fontsize=font_size, fontweight='bold')
            axes[1,0].legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            axes[1,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,0].grid(True, alpha=0.3)
            
            # Circuit cost distribution
            axes[1,1].hist(self.data['f2_circuit_cost'], bins=20, alpha=0.7, edgecolor='black', color='purple')
            axes[1,1].axvline(self.data['f2_circuit_cost'].mean(), color='red', linestyle='--',
                             label=f'Mean: {self.data["f2_circuit_cost"].mean():.2f}')
            axes[1,1].set_xlabel('Circuit Cost', fontsize=font_size, fontweight='bold')
            axes[1,1].set_ylabel('Frequency', fontsize=font_size, fontweight='bold')
            axes[1,1].legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            axes[1,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"{output_dir}/performance_distributions.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def plot_correlation_heatmap(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot correlation heatmap of numerical variables."""
        with sns.axes_style("whitegrid"):
            numerical_cols = ['val_acc', 'val_loss', 'f1_1_minus_acc', 'f2_circuit_cost', 
                             'f3_n_subcircuits', 'seconds', 'n_qubits', 'depth', 'learning_rate']
            
            correlation_matrix = self.data[numerical_cols].corr()
            
            fig = plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.3f', cbar_kws={"shrink": .8}, mask=mask,
                       annot_kws={'fontsize': font_size-2})
            plt.xticks(fontsize=font_size-1, fontweight='bold')
            plt.yticks(fontsize=font_size-1, fontweight='bold')
            plt.tight_layout()
            filename = f"{output_dir}/correlation_heatmap.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def plot_f1_f3_correlation(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot detailed correlation analysis between F1 and F3 as separate plots."""
        with sns.axes_style("whitegrid"):
            # Calculate correlation coefficients
            corr_f1_f3_pearson = self.data['f1_1_minus_acc'].corr(self.data['f3_n_subcircuits'])
            corr_f1_f3_spearman = self.data['f1_1_minus_acc'].corr(self.data['f3_n_subcircuits'], method='spearman')
            corr_f1_f3_kendall = self.data['f1_1_minus_acc'].corr(self.data['f3_n_subcircuits'], method='kendall')
            
            # Helper to save figure
            def save_fig(filename_base):
                filename = f"{output_dir}/{filename_base}.{file_ext}"
                if file_ext == "svg":
                    plt.savefig(filename, format='svg', bbox_inches='tight')
                else:
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 1. Direct F1 vs F3 scatter plot with trend line
            plt.figure(figsize=(10, 8))
            scatter1 = plt.scatter(self.data['f1_1_minus_acc'], self.data['f3_n_subcircuits'], 
                                  c=self.data['val_acc'], cmap='RdYlGn', alpha=0.7, s=60)
            plt.xlabel('F1: 1-Accuracy (minimize)', fontsize=font_size, fontweight='bold')
            plt.ylabel('F3: Number of Subcircuits (minimize)', fontsize=font_size, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=font_size-1)
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(self.data['f1_1_minus_acc'], self.data['f3_n_subcircuits'], 1)
            p = np.poly1d(z)
            plt.plot(self.data['f1_1_minus_acc'], p(self.data['f1_1_minus_acc']), "r--", alpha=0.8, linewidth=2, label='Trend line')
            
            cbar1 = plt.colorbar(scatter1)
            cbar1.set_label('Validation Accuracy (%)', fontsize=font_size-1)
            cbar1.ax.tick_params(labelsize=font_size-2)
            plt.legend()
            plt.tight_layout()
            save_fig('f1_f3_scatter_trend')
            
            # 2. F1 vs F3 colored by generation
            plt.figure(figsize=(10, 8))
            scatter2 = plt.scatter(self.data['f1_1_minus_acc'], self.data['f3_n_subcircuits'], 
                                  c=self.data['generation'], cmap='viridis', alpha=0.7, s=60)
            plt.xlabel('F1: 1-Accuracy (minimize)', fontsize=font_size, fontweight='bold')
            plt.ylabel('F3: Number of Subcircuits (minimize)', fontsize=font_size, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=font_size-1)
            plt.grid(True, alpha=0.3)
            
            cbar2 = plt.colorbar(scatter2)
            cbar2.set_label('Generation', fontsize=font_size-1)
            cbar2.ax.tick_params(labelsize=font_size-2)
            plt.tight_layout()
            save_fig('f1_f3_evolution')
            
            # 3. F1 distribution by F3 groups (box plot)
            plt.figure(figsize=(12, 8))
            f3_unique = sorted(self.data['f3_n_subcircuits'].unique())
            f1_by_f3 = [self.data[self.data['f3_n_subcircuits'] == f3]['f1_1_minus_acc'] for f3 in f3_unique]
            
            bp = plt.boxplot(f1_by_f3, labels=[f'F3={int(f3)}' for f3 in f3_unique], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            plt.xlabel('F3 Groups', fontsize=font_size, fontweight='bold')
            plt.ylabel('F1: 1-Accuracy', fontsize=font_size, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=font_size-1)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig('f1_f3_distribution')
            
            # 4. Joint density plot
            plt.figure(figsize=(10, 8))
            plt.hexbin(self.data['f1_1_minus_acc'], self.data['f3_n_subcircuits'], 
                      gridsize=20, cmap='Blues', alpha=0.8)
            plt.xlabel('F1: 1-Accuracy (minimize)', fontsize=font_size, fontweight='bold')
            plt.ylabel('F3: Number of Subcircuits (minimize)', fontsize=font_size, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=font_size-1)
            plt.grid(True, alpha=0.3)
            
            # Add correlation statistics text
            stats_text = f'''Correlation Statistics:
Pearson ρ: {corr_f1_f3_pearson:.3f}
Spearman ρ: {corr_f1_f3_spearman:.3f}
Kendall τ: {corr_f1_f3_kendall:.3f}'''
            
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    fontsize=font_size-1, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.colorbar(label='Density')
            plt.tight_layout()
            save_fig('f1_f3_density')
            
            print(f"\nF1-F3 CORRELATION ANALYSIS:")
            print(f"   Pearson correlation: {corr_f1_f3_pearson:.3f}")
            print(f"   Spearman correlation: {corr_f1_f3_spearman:.3f}")
            print(f"   Kendall tau: {corr_f1_f3_kendall:.3f}")
            print(f"[OK] F1-F3 correlation plots saved as {file_ext.upper()}:")
            print(f"   - {output_dir}/f1_f3_scatter_trend.{file_ext}")
            print(f"   - {output_dir}/f1_f3_evolution.{file_ext}")
            print(f"   - {output_dir}/f1_f3_distribution.{file_ext}")
            print(f"   - {output_dir}/f1_f3_density.{file_ext}")
    
    def plot_3d_accuracy_f2_f3(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Create separate 3D visualizations showing accuracy, F2, and F3 relationships."""
        from mpl_toolkits.mplot3d import Axes3D
        
        with sns.axes_style("whitegrid"):
            # 1. 3D scatter colored by generation
            fig = plt.figure(figsize=(12, 9))
            ax1 = fig.add_subplot(111, projection='3d')
            
            scatter1 = ax1.scatter(self.data['val_acc'], self.data['f2_circuit_cost'], 
                                  self.data['f3_n_subcircuits'], c=self.data['generation'], 
                                  cmap='viridis', alpha=0.7, s=60)
            ax1.set_xlabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
            ax1.set_ylabel('F2: Circuit Cost', fontsize=font_size, fontweight='bold', labelpad=15)
            ax1.set_zlabel('F3: Subcircuits', fontsize=font_size, fontweight='bold', labelpad=15)
            
            cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
            cbar1.set_label('Generation', fontsize=font_size-1)
            
            # Helper to save figure
            def save_fig(filename_base):
                filename = f"{output_dir}/{filename_base}.{file_ext}"
                if file_ext == "svg":
                    plt.savefig(filename, format='svg', bbox_inches='tight')
                else:
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            plt.tight_layout()
            save_fig('3d_accuracy_f2_f3_generations')
            
            # 2. 3D scatter colored by training time
            fig = plt.figure(figsize=(12, 9))
            ax2 = fig.add_subplot(111, projection='3d')
            
            scatter2 = ax2.scatter(self.data['val_acc'], self.data['f2_circuit_cost'], 
                                  self.data['f3_n_subcircuits'], c=self.data['seconds'], 
                                  cmap='plasma', alpha=0.7, s=60)
            ax2.set_xlabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
            ax2.set_ylabel('F2: Circuit Cost', fontsize=font_size, fontweight='bold', labelpad=15)
            ax2.set_zlabel('F3: Subcircuits', fontsize=font_size, fontweight='bold', labelpad=15)
            
            cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
            cbar2.set_label('Training Time (s)', fontsize=font_size-1)
            plt.tight_layout()
            save_fig('3d_accuracy_f2_f3_training_time')
            
            # 3. 3D scatter colored by embedding type
            fig = plt.figure(figsize=(12, 9))
            ax3 = fig.add_subplot(111, projection='3d')
            
            embed_colors = {'amplitude': 0, 'angle-x': 1, 'angle-y': 2, 'angle-z': 3}
            embed_numeric = self.data['embed'].map(embed_colors).fillna(0)
            scatter3 = ax3.scatter(self.data['val_acc'], self.data['f2_circuit_cost'], 
                                  self.data['f3_n_subcircuits'], c=embed_numeric, 
                                  cmap='Set1', alpha=0.7, s=60)
            ax3.set_xlabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
            ax3.set_ylabel('F2: Circuit Cost', fontsize=font_size, fontweight='bold', labelpad=15)
            ax3.set_zlabel('F3: Subcircuits', fontsize=font_size, fontweight='bold', labelpad=15)
            
            # Create custom legend for embedding types
            unique_embeds = self.data['embed'].unique()
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.Set1(embed_colors.get(embed, 0)), 
                                        markersize=8, label=embed) for embed in unique_embeds if embed in embed_colors]
            ax3.legend(handles=legend_elements, loc='upper right')
            plt.tight_layout()
            save_fig('3d_accuracy_f2_f3_embeddings')
            
            # 4. 3D Pareto front visualization
            fig = plt.figure(figsize=(12, 9))
            ax4 = fig.add_subplot(111, projection='3d')
            
            # Find 3D Pareto front
            objectives = self.data[['f1_1_minus_acc', 'f2_circuit_cost', 'f3_n_subcircuits']].values
            pareto_indices = self._find_pareto_front(objectives)
            
            # Plot all points in light color
            ax4.scatter(self.data['val_acc'], self.data['f2_circuit_cost'], 
                       self.data['f3_n_subcircuits'], c='lightblue', alpha=0.4, s=30, label='All solutions')
            
            # Highlight Pareto optimal points
            pareto_data = self.data.iloc[pareto_indices]
            ax4.scatter(pareto_data['val_acc'], pareto_data['f2_circuit_cost'], 
                       pareto_data['f3_n_subcircuits'], c='red', 
                       s=100, alpha=0.9, marker='*', label='Pareto optimal')
            
            # Also highlight the best accuracy point
            best_acc_idx = self.data['val_acc'].idxmax()
            best_point = self.data.loc[best_acc_idx]
            ax4.scatter([best_point['val_acc']], [best_point['f2_circuit_cost']], 
                       [best_point['f3_n_subcircuits']], c='gold', s=150, 
                       marker='D', alpha=1.0, edgecolors='black', linewidth=2, label='Best accuracy')
            
            ax4.set_xlabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
            ax4.set_ylabel('F2: Circuit Cost', fontsize=font_size, fontweight='bold', labelpad=15)
            ax4.set_zlabel('F3: Subcircuits', fontsize=font_size, fontweight='bold', labelpad=15)
            ax4.legend()
            plt.tight_layout()
            save_fig('3d_accuracy_f2_f3_pareto')
            
            # Print 3D analysis summary
            print(f"\n🎯 3D ANALYSIS SUMMARY:")
            print(f"   Total solutions: {len(self.data)}")
            print(f"   Pareto optimal solutions: {len(pareto_indices)}")
            print(f"   Best accuracy: {self.data['val_acc'].max():.2f}%")
            print(f"   Best accuracy config: F2={int(best_point['f2_circuit_cost'])}, F3={int(best_point['f3_n_subcircuits'])}")
            print(f"   Accuracy range: {self.data['val_acc'].min():.1f}% - {self.data['val_acc'].max():.1f}%")
            print(f"   F2 range: {int(self.data['f2_circuit_cost'].min())} - {int(self.data['f2_circuit_cost'].max())}")
            print(f"   F3 range: {int(self.data['f3_n_subcircuits'].min())} - {int(self.data['f3_n_subcircuits'].max())}")
            print(f"[OK] 3D plots saved as {file_ext.upper()}:")
            print(f"   - {output_dir}/3d_accuracy_f2_f3_generations.{file_ext}")
            print(f"   - {output_dir}/3d_accuracy_f2_f3_training_time.{file_ext}")
            print(f"   - {output_dir}/3d_accuracy_f2_f3_embeddings.{file_ext}")
            print(f"   - {output_dir}/3d_accuracy_f2_f3_pareto.{file_ext}")
    
    def plot_best_performers(self, output_dir="outputs/figures", top_n=10, font_size=12, file_ext="png"):
        """Analyze and visualize the best performing configurations."""
        with sns.axes_style("whitegrid"):
            # Get top performers by accuracy
            top_performers = self.data.nlargest(top_n, 'val_acc')
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Best accuracy configurations
            axes[0,0].barh(range(len(top_performers)), top_performers['val_acc'])
            axes[0,0].set_yticks(range(len(top_performers)))
            axes[0,0].set_yticklabels([f"Eval {i+1}" for i in range(len(top_performers))])
            axes[0,0].set_xlabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[0,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0,0].invert_yaxis()
            axes[0,0].grid(True, alpha=0.3)
            
            # Embedding type distribution in top performers
            embed_counts = top_performers['embed'].value_counts()
            axes[0,1].pie(embed_counts.values, labels=embed_counts.index, autopct='%1.1f%%',
                         textprops={'fontsize': font_size-1, 'fontweight': 'bold'})
            
            # Qubit distribution in top performers
            qubit_counts = top_performers['n_qubits'].value_counts().sort_index()
            axes[1,0].bar(qubit_counts.index, qubit_counts.values)
            axes[1,0].set_xlabel('Number of Qubits', fontsize=font_size, fontweight='bold')
            axes[1,0].set_ylabel('Count', fontsize=font_size, fontweight='bold')
            axes[1,0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,0].grid(True, alpha=0.3)
            
            # Accuracy vs efficiency for top performers
            axes[1,1].scatter(top_performers['efficiency'], top_performers['val_acc'], 
                             s=100, alpha=0.7, c='red')
            axes[1,1].set_xlabel('Efficiency (acc/second)', fontsize=font_size, fontweight='bold')
            axes[1,1].set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            axes[1,1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"{output_dir}/best_performers.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Print summary of best performers
            print(f"\n=== TOP {top_n} PERFORMERS SUMMARY ===")
            print(top_performers[['eval_id', 'embed', 'n_qubits', 'depth', 'val_acc', 'val_loss', 'seconds']].to_string(index=False))
    
    def plot_comprehensive_summary(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Create a comprehensive summary dashboard."""
        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Evolution timeline
        ax1 = fig.add_subplot(gs[0, :2])
        gen_stats = self.data.groupby('generation')['val_acc'].agg(['mean', 'max'])
        ax1.plot(gen_stats.index, gen_stats['mean'], 'o-', label='Mean Accuracy', linewidth=2)
        ax1.plot(gen_stats.index, gen_stats['max'], 's-', label='Best Accuracy', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pareto front
        ax2 = fig.add_subplot(gs[0, 2:])
        scatter = ax2.scatter(self.data['f2_circuit_cost'], self.data['val_acc'], 
                             c=self.data['generation'], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('Circuit Cost')
        ax2.set_ylabel('Accuracy (%)')
        plt.colorbar(scatter, ax=ax2, label='Generation')
        
        # 3. Hyperparameter impact
        ax3 = fig.add_subplot(gs[1, 0])
        embed_stats = self.data.groupby('embed')['val_acc'].mean()
        ax3.bar(embed_stats.index, embed_stats.values)
        ax3.set_ylabel('Avg Accuracy (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        ax4 = fig.add_subplot(gs[1, 1])
        qubit_stats = self.data.groupby('n_qubits')['val_acc'].mean()
        ax4.plot(qubit_stats.index, qubit_stats.values, 'o-', linewidth=2)
        ax4.set_xlabel('Number of Qubits')
        ax4.set_ylabel('Avg Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        
        # 4. Performance distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(self.data['val_acc'], bins=15, alpha=0.7, edgecolor='black')
        ax5.axvline(self.data['val_acc'].mean(), color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Accuracy (%)')
        ax5.set_ylabel('Frequency')
        
        # 5. Training efficiency
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.scatter(self.data['seconds'], self.data['val_acc'], alpha=0.6)
        ax6.set_xlabel('Training Time (s)')
        ax6.set_ylabel('Accuracy (%)')
        
        # 6. Statistics summary
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Calculate key statistics
        stats_text = f"""
        OPTIMIZATION SUMMARY STATISTICS
        
        Total Evaluations: {len(self.data)}
        Generations: {self.data['generation'].max() + 1}
        
        ACCURACY METRICS:
        Best Accuracy: {self.data['val_acc'].max():.2f}%
        Mean Accuracy: {self.data['val_acc'].mean():.2f}%
        Std Accuracy: {self.data['val_acc'].std():.2f}%
        
        EFFICIENCY METRICS:
        Best Training Time: {self.data['seconds'].min():.2f}s
        Mean Training Time: {self.data['seconds'].mean():.2f}s
        Best Efficiency: {self.data['efficiency'].max():.4f} acc/s
        
        CIRCUIT COMPLEXITY:
        Min Circuit Cost: {self.data['f2_circuit_cost'].min():.0f}
        Max Circuit Cost: {self.data['f2_circuit_cost'].max():.0f}
        Qubit Range: {self.data['n_qubits'].min()}-{self.data['n_qubits'].max()}
        Depth Range: {self.data['depth'].min()}-{self.data['depth'].max()}
        
        BEST CONFIGURATION:
        Eval ID: {self.data.loc[self.data['val_acc'].idxmax(), 'eval_id']}
        Embedding: {self.data.loc[self.data['val_acc'].idxmax(), 'embed']}
        Qubits: {self.data.loc[self.data['val_acc'].idxmax(), 'n_qubits']}
        Depth: {self.data.loc[self.data['val_acc'].idxmax(), 'depth']}
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        filename = f"{output_dir}/comprehensive_dashboard.{file_ext}"
        if file_ext == "svg":
            plt.savefig(filename, format='svg', bbox_inches='tight')
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    

    def plot_f3_paper(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Single comprehensive F3 analysis figure for research paper.
        
        Creates a single plot supporting the cut-aware analysis paragraph:
        - Most models used F3=1
        - Configurations with n≥5 required cutting (F3>1)
        - Correlations: F2 vs time (ρ=0.94), F3 vs time (ρ=0.86), accuracy vs F3 (ρ=0.12)
        """
        with sns.axes_style("whitegrid"):
            output_dir = self.create_output_dir(output_dir)
            
            # Calculate correlations
            corr_f2_time = self.data['f2_circuit_cost'].corr(self.data['seconds'])
            corr_f3_time = self.data['f3_n_subcircuits'].corr(self.data['seconds'])
            corr_acc_f3 = self.data['val_acc'].corr(self.data['f3_n_subcircuits'])
            
            # Create single comprehensive plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Main scatter: F3 vs n_qubits colored by training time
            scatter = ax.scatter(self.data['n_qubits'], self.data['f3_n_subcircuits'], 
                               c=self.data['seconds'], cmap='plasma', alpha=0.7, s=60,
                               edgecolors='black', linewidth=0.5)
            
            # Add vertical line at cut target to show cutting threshold
            cut_target = CUT_TARGET_QUBITS if CUT_TARGET_QUBITS > 0 else 5
            ax.axvline(x=cut_target, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                      label=f'n≥{cut_target} requires cutting (F₃>1)')
            
            # Highlight F3=1 region
            f3_1_mask = self.data['f3_n_subcircuits'] == 1
            ax.scatter(self.data[f3_1_mask]['n_qubits'], self.data[f3_1_mask]['f3_n_subcircuits'],
                      facecolors='none', edgecolors='blue', linewidth=2, s=80, alpha=0.8,
                      label='F₃=1 (no cutting)')
            
            ax.set_xlabel('Number of Qubits (n)', fontsize=font_size+2, fontweight='bold')
            ax.set_ylabel('Number of Subcircuits (F₃)', fontsize=font_size+2, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=font_size, frameon=True, fancybox=True, shadow=True)
            
            # Add colorbar for training time
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Time (seconds)', fontsize=font_size, fontweight='bold')
            cbar.ax.tick_params(labelsize=font_size-1)
            
            # Add text box with key statistics and correlations
            f3_1_count = (self.data['f3_n_subcircuits'] == 1).sum()
            total_count = len(self.data)
            f3_1_percent = (f3_1_count / total_count) * 100
            
            stats_text = f'''Key Findings:
• F₃=1: {f3_1_percent:.1f}% of models
• F₂ ↔ Time: ρ = {corr_f2_time:.2f}
• F₃ ↔ Time: ρ = {corr_f3_time:.2f}  
• Acc ↔ F₃: ρ = {corr_acc_f3:.2f}'''
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=font_size,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor='lightblue', alpha=0.9), fontweight='bold')
            
            plt.tight_layout()
            filename = f"{output_dir}/f3_paper.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\nF3 PAPER FIGURE GENERATED:")
            print(f"   Supporting evidence for cut-aware analysis paragraph")
            print(f"   Key correlations: F₂↔Time({corr_f2_time:.2f}), F₃↔Time({corr_f3_time:.2f}), Acc↔F₃({corr_acc_f3:.2f})")
            print(f"[OK] Single F3 figure saved to {filename}")

    def plot_f3_analysis(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Cut-aware analysis (F3) - Figure for research paper.
        
        Analyzes the number of subcircuits (F3) showing:
        1. Most models used F3=1
        2. Configurations with n≥5 qubits required cutting (F3>1)
        3. Correlation analysis: F2 vs time (ρ=0.94), F3 vs time (ρ=0.86), accuracy vs F3 (ρ=0.12)
        """
        with sns.axes_style("whitegrid"):
            # Use output_dir as provided (already processed by caller)
            
            # Calculate correlations
            corr_f2_time = self.data['f2_circuit_cost'].corr(self.data['seconds'])
            corr_f3_time = self.data['f3_n_subcircuits'].corr(self.data['seconds'])
            corr_acc_f3 = self.data['val_acc'].corr(self.data['f3_n_subcircuits'])
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. F3 distribution showing most models used F3=1
            f3_counts = self.data['f3_n_subcircuits'].value_counts().sort_index()
            axes[0, 0].bar(f3_counts.index, f3_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Number of Subcircuits (F₃)', fontsize=font_size, fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontsize=font_size, fontweight='bold')
            axes[0, 0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add text annotation showing F3=1 dominance
            f3_1_count = (self.data['f3_n_subcircuits'] == 1).sum()
            total_count = len(self.data)
            f3_1_percent = (f3_1_count / total_count) * 100
            axes[0, 0].text(0.7, 0.9, f'F₃=1: {f3_1_percent:.1f}% of models', 
                           transform=axes[0, 0].transAxes, fontsize=font_size-1, 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 2. F3 vs number of qubits (showing n≥5 requires cutting)
            scatter1 = axes[0, 1].scatter(self.data['n_qubits'], self.data['f3_n_subcircuits'], 
                                         alpha=0.7, s=50, c=self.data['val_acc'], cmap='viridis')
            axes[0, 1].set_xlabel('Number of Qubits (n)', fontsize=font_size, fontweight='bold')
            axes[0, 1].set_ylabel('Number of Subcircuits (F₃)', fontsize=font_size, fontweight='bold')
            axes[0, 1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add vertical line at cut target to show cutting threshold
            cut_target = CUT_TARGET_QUBITS if CUT_TARGET_QUBITS > 0 else 5
            axes[0, 1].axvline(x=cut_target, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'n≥{cut_target} requires cutting')
            axes[0, 1].legend(fontsize=font_size-1)
            
            # Add colorbar for accuracy
            cbar1 = plt.colorbar(scatter1, ax=axes[0, 1])
            cbar1.set_label('Validation Accuracy (%)', fontsize=font_size-1)
            cbar1.ax.tick_params(labelsize=font_size-2)
            
            # 3. F2 vs Training Time (ρ=0.94)
            axes[1, 0].scatter(self.data['f2_circuit_cost'], self.data['seconds'], 
                              alpha=0.6, s=50, color='orange')
            axes[1, 0].set_xlabel('Circuit Cost (F₂)', fontsize=font_size, fontweight='bold')
            axes[1, 0].set_ylabel('Training Time (seconds)', fontsize=font_size, fontweight='bold')
            axes[1, 0].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. F3 vs Training Time (ρ=0.86) and Accuracy vs F3 (ρ=0.12)
            scatter2 = axes[1, 1].scatter(self.data['f3_n_subcircuits'], self.data['seconds'], 
                                         alpha=0.6, s=50, c=self.data['val_acc'], cmap='plasma')
            axes[1, 1].set_xlabel('Number of Subcircuits (F₃)', fontsize=font_size, fontweight='bold')
            axes[1, 1].set_ylabel('Training Time (seconds)', fontsize=font_size, fontweight='bold')
            axes[1, 1].tick_params(axis='both', which='major', labelsize=font_size-1)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add text annotation for accuracy vs F3 correlation
            axes[1, 1].text(0.05, 0.95, f'Accuracy vs F₃: ρ = {corr_acc_f3:.2f}', 
                           transform=axes[1, 1].transAxes, fontsize=font_size-1,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # Add colorbar for accuracy
            cbar2 = plt.colorbar(scatter2, ax=axes[1, 1])
            cbar2.set_label('Validation Accuracy (%)', fontsize=font_size-1)
            cbar2.ax.tick_params(labelsize=font_size-2)
            
            plt.tight_layout()
            filename = f"{output_dir}/f3_comprehensive_analysis.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Print correlation summary
            print(f"\nCUT-AWARE ANALYSIS (F₃) SUMMARY:")
            print(f"   F₃ = 1: {f3_1_percent:.1f}% of models")
            print(f"   F₂ vs Training Time: ρ = {corr_f2_time:.2f}")
            print(f"   F₃ vs Training Time: ρ = {corr_f3_time:.2f}")
            print(f"   Accuracy vs F₃: ρ = {corr_acc_f3:.2f}")
            print(f"[OK] F3 analysis saved to {filename}")

    def plot_pareto_front_f3(self, output_dir="outputs/figures", file_ext="png"):
        """Plot Pareto fronts involving F3."""
        # Use output_dir as provided (already processed by caller)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1 vs F3 Pareto front
        axes[0, 0].scatter(self.data['f1_1_minus_acc'], self.data['f3_n_subcircuits'], 
                          c=self.data['generation'], cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('F1: 1 - Accuracy (minimize)')
        axes[0, 0].set_ylabel('F3: Number of Subcircuits (minimize)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add Pareto front estimation for F1 vs F3
        pareto_f1_f3 = self._find_pareto_front(self.data[['f1_1_minus_acc', 'f3_n_subcircuits']].values)
        pareto_data_f1_f3 = self.data.iloc[pareto_f1_f3]
        axes[0, 0].scatter(pareto_data_f1_f3['f1_1_minus_acc'], pareto_data_f1_f3['f3_n_subcircuits'], 
                          color='red', s=100, marker='*', label='Pareto Front', zorder=5)
        axes[0, 0].legend()
        
        # 2. F2 vs F3 Pareto front
        axes[0, 1].scatter(self.data['f2_circuit_cost'], self.data['f3_n_subcircuits'], 
                          c=self.data['val_acc'], cmap='RdYlBu', alpha=0.7)
        axes[0, 1].set_xlabel('F2: Circuit Cost (minimize)')
        axes[0, 1].set_ylabel('F3: Number of Subcircuits (minimize)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add Pareto front for F2 vs F3
        pareto_f2_f3 = self._find_pareto_front(self.data[['f2_circuit_cost', 'f3_n_subcircuits']].values)
        pareto_data_f2_f3 = self.data.iloc[pareto_f2_f3]
        axes[0, 1].scatter(pareto_data_f2_f3['f2_circuit_cost'], pareto_data_f2_f3['f3_n_subcircuits'], 
                          color='red', s=100, marker='*', label='Pareto Front', zorder=5)
        axes[0, 1].legend()
        
        # 3. 3D Pareto front: F1, F2, F3
        ax3d = fig.add_subplot(2, 2, 3, projection='3d')
        scatter = ax3d.scatter(self.data['f1_1_minus_acc'], self.data['f2_circuit_cost'], 
                              self.data['f3_n_subcircuits'], c=self.data['generation'], 
                              cmap='viridis', alpha=0.6)
        
        # Highlight 3D Pareto front
        pareto_3d = self._find_pareto_front(self.data[['f1_1_minus_acc', 'f2_circuit_cost', 'f3_n_subcircuits']].values)
        pareto_data_3d = self.data.iloc[pareto_3d]
        ax3d.scatter(pareto_data_3d['f1_1_minus_acc'], pareto_data_3d['f2_circuit_cost'], 
                    pareto_data_3d['f3_n_subcircuits'], color='red', s=100, marker='*')
        
        ax3d.set_xlabel('F1: 1-Accuracy')
        ax3d.set_ylabel('F2: Circuit Cost')
        ax3d.set_zlabel('F3: Subcircuits')
        
        # 4. F3 efficiency analysis
        self.data['f3_efficiency'] = self.data['val_acc'] / self.data['f3_n_subcircuits']
        axes[1, 1].scatter(self.data['f3_n_subcircuits'], self.data['f3_efficiency'], 
                          c=self.data['f2_circuit_cost'], cmap='plasma', alpha=0.7)
        axes[1, 1].set_xlabel('Number of Subcircuits (F3)')
        axes[1, 1].set_ylabel('Accuracy / F3 (Efficiency)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{output_dir}/pareto_fronts_with_f3.{file_ext}"
        if file_ext == "svg":
            plt.savefig(filename, format='svg', bbox_inches='tight')
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _find_pareto_front(self, costs):
        """Find Pareto front for minimization problems."""
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

    def plot_f3_detailed_analysis(self, output_dir="outputs/figures", file_ext="png"):
        """Detailed analysis focusing specifically on F3 characteristics."""
        # Use output_dir as provided (already processed by caller)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. F3 vs all parameters
        params = ['n_qubits', 'depth', 'learning_rate', 'val_loss', 'seconds']
        param_labels = ['Number of Qubits', 'Circuit Depth', 'Learning Rate', 'Validation Loss', 'Training Time']
        
        for i, (param, label) in enumerate(zip(params[:2], param_labels[:2])):
            if param in self.data.columns:
                scatter = axes[0, i].scatter(self.data[param], self.data['f3_n_subcircuits'], 
                                           c=self.data['val_acc'], cmap='RdYlGn', alpha=0.7)
                axes[0, i].set_xlabel(label)
                axes[0, i].set_ylabel('Number of Subcircuits (F3)')
                axes[0, i].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, i], label='Validation Accuracy')
        
        # 2. F3 correlation with performance metrics
        perf_metrics = ['val_acc', 'val_loss']
        for i, metric in enumerate(perf_metrics):
            if metric in self.data.columns:
                axes[1, i].scatter(self.data['f3_n_subcircuits'], self.data[metric], 
                                 c=self.data['n_qubits'], cmap='viridis', alpha=0.7)
                axes[1, i].set_xlabel('Number of Subcircuits (F3)')
                axes[1, i].set_ylabel(metric.replace("_", " ").title())
                axes[1, i].grid(True, alpha=0.3)
        
        # 3. F3 statistics by configuration
        # Group by quantum configuration
        if all(col in self.data.columns for col in ['n_qubits', 'depth']):
            self.data['config'] = self.data['n_qubits'].astype(str) + 'q_' + self.data['depth'].astype(str) + 'd'
            config_stats = self.data.groupby('config')['f3_n_subcircuits'].agg(['mean', 'std', 'count'])
            
            axes[2, 0].bar(range(len(config_stats)), config_stats['mean'], 
                          yerr=config_stats['std'], capsize=5, alpha=0.7)
            axes[2, 0].set_xlabel('Configuration (Qubits_Depth)')
            axes[2, 0].set_ylabel('Average Number of Subcircuits')
            axes[2, 0].set_xticks(range(len(config_stats)))
            axes[2, 0].set_xticklabels(config_stats.index, rotation=45)
            axes[2, 0].grid(True, alpha=0.3)
        
        # 4. F3 optimal regions
        # Find regions where F3 gives best trade-offs
        self.data['composite_score'] = self.data['val_acc'] / (self.data['f3_n_subcircuits'] + 1)  # +1 to avoid division by zero
        
        scatter = axes[2, 1].scatter(self.data['f3_n_subcircuits'], self.data['composite_score'], 
                                   c=self.data['val_acc'], cmap='RdYlGn', alpha=0.7, s=60)
        
        # Highlight top performers
        top_performers = self.data.nlargest(5, 'composite_score')
        axes[2, 1].scatter(top_performers['f3_n_subcircuits'], top_performers['composite_score'], 
                          color='red', s=100, marker='*', label='Top 5 Performers', zorder=5)
        
        axes[2, 1].set_xlabel('Number of Subcircuits (F3)')
        axes[2, 1].set_ylabel('Composite Score (Accuracy/F3)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        plt.colorbar(scatter, ax=axes[2, 1], label='Validation Accuracy')
        
        plt.tight_layout()
        filename = f"{output_dir}/f3_detailed_analysis.{file_ext}"
        if file_ext == "svg":
            plt.savefig(filename, format='svg', bbox_inches='tight')
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Print F3 statistics
        print("\n" + "="*50)
        print("F3 (Number of Subcircuits) Statistics")
        print("="*50)
        print(f"Mean F3: {self.data['f3_n_subcircuits'].mean():.2f}")
        print(f"Std F3: {self.data['f3_n_subcircuits'].std():.2f}")
        print(f"Min F3: {self.data['f3_n_subcircuits'].min()}")
        print(f"Max F3: {self.data['f3_n_subcircuits'].max()}")
        print(f"Median F3: {self.data['f3_n_subcircuits'].median():.2f}")
        
        print(f"\nBest performing configuration with lowest F3:")
        min_f3_best = self.data[self.data['f3_n_subcircuits'] == self.data['f3_n_subcircuits'].min()]
        best_min_f3 = min_f3_best.loc[min_f3_best['val_acc'].idxmax()]
        print(f"F3: {best_min_f3['f3_n_subcircuits']}, Accuracy: {best_min_f3['val_acc']:.2f}%")
        print(f"Configuration: {best_min_f3['n_qubits']}q, depth {best_min_f3['depth']}, {best_min_f3['embed']}")

    def plot_pareto_tradeoffs_highlighted(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot Pareto trade-offs highlighting the best search-phase accuracy of 24.0%.
        
        This plot specifically shows the multi-objective landscape with emphasis on
        the best performing configuration for research paper figures.
        
        Parameters:
        -----------
        output_dir : str, default="plots"
            Directory to save the plot
        font_size : int, default=12
            Base font size for all text elements
        file_ext : str, default="png"
            File extension for saved plots ("png" or "svg")
        """
        # Find the best performing solution
        best_idx = self.data['val_acc'].idxmax()
        best_solution = self.data.loc[best_idx]
        
        # Create the main Pareto plot (single figure - accuracy vs circuit cost only)
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot: Accuracy vs Circuit Cost (main trade-off)
            # All solutions
            scatter = ax.scatter(self.data['f2_circuit_cost'], self.data['val_acc'], 
                               c=self.data['gen_est'], cmap='viridis', alpha=0.6, s=50,
                               label='All Solutions')
            
            # Highlight the best solution (24.0% accuracy)
            ax.scatter(best_solution['f2_circuit_cost'], best_solution['val_acc'], 
                      color='red', s=200, marker='*', zorder=10, 
                      label=f'Best: {best_solution["val_acc"]}% (Gen {int(best_solution["gen_est"])})')
            
            # Add Pareto front
            pareto_data = self.data[['f2_circuit_cost', 'val_acc']].values
            # Convert to minimization problem (negate accuracy)
            pareto_data_min = pareto_data.copy()
            pareto_data_min[:, 1] = -pareto_data_min[:, 1]
            pareto_indices = self._find_pareto_front(pareto_data_min)
            pareto_solutions = self.data.iloc[pareto_indices].sort_values('f2_circuit_cost')
            
            ax.plot(pareto_solutions['f2_circuit_cost'], pareto_solutions['val_acc'], 
                   'r--', linewidth=2, alpha=0.8, label='Pareto Front')
            
            ax.set_xlabel('Circuit Cost (F2)', fontsize=font_size, fontweight='bold')
            ax.set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
            ax.legend(fontsize=font_size-1, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=font_size-1)
            
            # Add colorbar for generation
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Generation', fontsize=font_size-1)
            cbar.ax.tick_params(labelsize=font_size-2)
            
            plt.tight_layout()
            filename = f"{output_dir}/pareto_tradeoffs_highlighted.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # Print best solution details
        print(f"\n🏆 BEST SEARCH-PHASE PERFORMANCE HIGHLIGHTED:")
        print(f"   Accuracy: {best_solution['val_acc']}% (Generation {int(best_solution['gen_est'])})")
        print(f"   Configuration: {int(best_solution['n_qubits'])}q-{int(best_solution['depth'])}d, {best_solution['embed']}")
        print(f"   Objectives: F1={best_solution['f1_1_minus_acc']:.3f}, F2={int(best_solution['f2_circuit_cost'])}, F3={int(best_solution['f3_n_subcircuits'])}")
        print(f"   Eval ID: {best_solution['eval_id']}")
        print(f"[OK] Plot saved: {filename}")

    def generate_f3_focused_plots(self, output_dir="outputs/figures", file_ext="png"):
        """Generate all F3-focused plots."""
        if self.data is None:
            print("No data loaded. Cannot generate plots.")
            return
        
        output_dir = self.create_output_dir(output_dir) if isinstance(output_dir, str) else str(output_dir)
        print(f"Generating F3-focused plots in directory: {output_dir}")
        
        print("1. Generating F3 comprehensive analysis...")
        self.plot_f3_analysis(output_dir, file_ext=file_ext)
        
        print("2. Generating Pareto fronts with F3...")
        self.plot_pareto_front_f3(output_dir, file_ext=file_ext)
        
        print("3. Generating detailed F3 analysis...")
        self.plot_f3_detailed_analysis(output_dir, file_ext=file_ext)
        
        print(f"\nAll F3-focused plots generated successfully in '{output_dir}' directory!")


    def print_data_summary(self):
        """Print comprehensive data summary and statistics."""
        if self.data is None:
            print("No data loaded.")
            return
        
        print("="*70)
        print("NSGA-II QUANTUM NEURAL NETWORK OPTIMIZATION - DATA SUMMARY")
        print("="*70)
        
        # Basic statistics
        print(f"\nDATASET OVERVIEW:")
        print(f"   Total evaluations: {len(self.data)}")
        print(f"   Generations: {self.data['gen_est'].min()} → {self.data['gen_est'].max()}")
        print(f"   Unique configurations: {len(self.data[['n_qubits', 'depth', 'embed']].drop_duplicates())}")
        
        # Performance statistics
        acc_stats = self.data['val_acc'].describe()
        print(f"\n🎯 PERFORMANCE STATISTICS:")
        print(f"   Accuracy range: {acc_stats['min']:.1f}% → {acc_stats['max']:.1f}%")
        print(f"   Mean ± Std: {acc_stats['mean']:.2f}% ± {acc_stats['std']:.2f}%")
        print(f"   Median: {acc_stats['50%']:.1f}%")
        
        # F1, F2, F3 statistics
        print(f"\n🎯 OBJECTIVE FUNCTIONS:")
        print(f"   F1 (1-Accuracy): {self.data['f1_1_minus_acc'].min():.3f} → {self.data['f1_1_minus_acc'].max():.3f}")
        print(f"   F2 (Circuit Cost): {self.data['f2_circuit_cost'].min()} → {self.data['f2_circuit_cost'].max()}")
        print(f"   F3 (Subcircuits): {self.data['f3_n_subcircuits'].min()} → {self.data['f3_n_subcircuits'].max()}")
        
        # Best configurations
        best_acc = self.data.loc[self.data['val_acc'].idxmax()]
        print(f"\n🏆 BEST PERFORMING CONFIGURATION:")
        print(f"   Eval ID: {best_acc['eval_id']}")
        print(f"   Accuracy: {best_acc['val_acc']:.1f}%")
        print(f"   Config: {best_acc['n_qubits']}q-{best_acc['depth']}d, {best_acc['embed']}")
        print(f"   F1={best_acc['f1_1_minus_acc']:.3f}, F2={best_acc['f2_circuit_cost']}, F3={best_acc['f3_n_subcircuits']}")
        print(f"   Training time: {best_acc['seconds']:.1f}s")
        
        # F3 insights
        f3_stats = self.data['f3_n_subcircuits'].describe()
        print(f"\n🔧 F3 (SUBCIRCUITS) INSIGHTS:")
        print(f"   Range: {f3_stats['min']:.0f} → {f3_stats['max']:.0f}")
        print(f"   Mean ± Std: {f3_stats['mean']:.2f} ± {f3_stats['std']:.2f}")
        print(f"   Most common: {self.data['f3_n_subcircuits'].mode()[0]} ({(self.data['f3_n_subcircuits'] == self.data['f3_n_subcircuits'].mode()[0]).sum()} occurrences)")
        
        # Correlations
        corr_f3_acc = self.data['f3_n_subcircuits'].corr(self.data['val_acc'])
        corr_f3_time = self.data['f3_n_subcircuits'].corr(self.data['seconds'])
        print(f"\n🔗 KEY CORRELATIONS:")
        print(f"   F3 ↔ Accuracy: {corr_f3_acc:.3f}")
        print(f"   F3 ↔ Training Time: {corr_f3_time:.3f}")
        
        # Configuration analysis
        print(f"\n⚛️  CONFIGURATION BREAKDOWN:")
        config_counts = self.data.groupby(['n_qubits', 'depth']).size()
        for (qubits, depth), count in config_counts.items():
            avg_acc = self.data[(self.data['n_qubits'] == qubits) & (self.data['depth'] == depth)]['val_acc'].mean()
            avg_f3 = self.data[(self.data['n_qubits'] == qubits) & (self.data['depth'] == depth)]['f3_n_subcircuits'].mean()
            print(f"   {qubits}q-{depth}d: {count} evals, avg acc={avg_acc:.1f}%, avg F3={avg_f3:.1f}")
        
        print("="*70)

    def analyze_f3_insights(self):
        """Detailed F3 analysis with insights."""
        if self.data is None:
            print("No data loaded.")
            return
        
        print("="*70)
        print("F3 (NUMBER OF SUBCIRCUITS) - DETAILED INSIGHTS")
        print("="*70)
        
        # F3 correlations
        correlation_acc = self.data['f3_n_subcircuits'].corr(self.data['val_acc'])
        correlation_loss = self.data['f3_n_subcircuits'].corr(self.data['val_loss'])
        correlation_time = self.data['f3_n_subcircuits'].corr(self.data['seconds'])
        
        print(f"\n🔗 F3 CORRELATIONS:")
        print(f"   F3 ↔ Accuracy: {correlation_acc:.3f} {'(negative = good!)' if correlation_acc < 0 else '(positive = concerning)'}")
        print(f"   F3 ↔ Loss: {correlation_loss:.3f} {'(positive = concerning)' if correlation_loss > 0 else '(negative = good!)'}")
        print(f"   F3 ↔ Training Time: {correlation_time:.3f} {'(positive = expected)' if correlation_time > 0 else '(negative = surprising!)'}")
        
        # Best F3 configurations
        min_f3_configs = self.data[self.data['f3_n_subcircuits'] == self.data['f3_n_subcircuits'].min()]
        best_min_f3 = min_f3_configs.loc[min_f3_configs['val_acc'].idxmax()]
        
        print(f"\n🏆 OPTIMAL F3 CONFIGURATIONS:")
        print(f"   Best with minimal F3 ({best_min_f3['f3_n_subcircuits']}):")
        print(f"     → {best_min_f3['eval_id']}: {best_min_f3['val_acc']:.1f}% accuracy")
        print(f"     → Config: {best_min_f3['n_qubits']}q-{best_min_f3['depth']}d, {best_min_f3['embed']}")
        
        # F3 efficiency
        self.data['f3_efficiency'] = self.data['val_acc'] / self.data['f3_n_subcircuits']
        best_efficiency = self.data.loc[self.data['f3_efficiency'].idxmax()]
        print(f"\n   Best F3 efficiency ({best_efficiency['f3_efficiency']:.2f} acc/subcircuit):")
        print(f"     → {best_efficiency['eval_id']}: {best_efficiency['val_acc']:.1f}% accuracy, F3 = {best_efficiency['f3_n_subcircuits']}")
        
        # Pareto analysis
        def is_pareto_efficient(costs):
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                    is_efficient[i] = True
            return is_efficient
        
        pareto_mask = is_pareto_efficient(self.data[['f1_1_minus_acc', 'f2_circuit_cost', 'f3_n_subcircuits']].values)
        pareto_solutions = self.data[pareto_mask]
        
        print(f"\n🎯 PARETO EFFICIENCY:")
        print(f"   Found {len(pareto_solutions)} Pareto-efficient solutions")
        print(f"   F3 range in Pareto front: {pareto_solutions['f3_n_subcircuits'].min()} → {pareto_solutions['f3_n_subcircuits'].max()}")
        
        print("="*70)

    def generate_all_plots(self, output_dir="outputs/figures", show_summary=False, font_size=12, file_ext="png"):
        """Generate all available plots with optional data summary.
        
        Parameters:
        -----------
        output_dir : str, default="outputs/figures"
            Directory to save all plots
        show_summary : bool, default=False
            Whether to show data summary before generating plots
        font_size : int, default=12
            Base font size for all plots
        file_ext : str, default="png"
            File extension for saved plots ("png" or "svg")
        """
        if self.data is None:
            print("No data loaded. Cannot generate plots.")
            return
        
        if show_summary:
            self.print_data_summary()
            print("\n")
        
        output_dir = self.create_output_dir(output_dir)
        print(f"🎨 GENERATING COMPREHENSIVE PLOT ANALYSIS")
        print(f"Output directory: {output_dir}")
        print("="*50)
        
        print("1️⃣  Generating evolution progress plots...")
        self.plot_evolution_progress(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("1️⃣b Generating generation summary plot...")
        self.plot_generation_summary(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("2️⃣  Generating Pareto front analysis...")
        self.plot_pareto_front(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("2️⃣b Generating Pareto trade-offs with highlight...")
        self.plot_pareto_tradeoffs_highlighted(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("3️⃣  Generating hyperparameter analysis...")
        self.plot_hyperparameter_analysis(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("4️⃣  Generating performance distributions...")
        self.plot_performance_distributions(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("5️⃣  Generating correlation heatmap...")
        self.plot_correlation_heatmap(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("6️⃣  Analyzing best performers...")
        self.plot_best_performers(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("7️⃣  Generating comprehensive dashboard...")
        self.plot_comprehensive_summary(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("8️⃣  Generating F3 comprehensive analysis...")
        self.plot_f3_analysis(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("9️⃣  Generating Pareto fronts with F3...")
        self.plot_pareto_front_f3(output_dir, file_ext=file_ext)
        
        print("🔟 Generating detailed F3 analysis...")
        self.plot_f3_detailed_analysis(output_dir, file_ext=file_ext)
        
        print("1️⃣1️⃣ Generating F1-F3 correlation analysis...")
        self.plot_f1_f3_correlation(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("1️⃣2️⃣ Generating 3D accuracy-F2-F3 visualization...")
        self.plot_3d_accuracy_f2_f3(output_dir, font_size=font_size, file_ext=file_ext)
        
        print("1️⃣3️⃣ Generating Pareto optimal training curves...")
        try:
            self.plot_pareto_training_curves(output_dir, font_size=font_size, file_ext=file_ext)
        except Exception as e:
            print(f"[WARN] Could not generate Pareto training curves: {e}")
        
        print("="*50)
        print(f"[OK] ALL PLOTS GENERATED SUCCESSFULLY!")
        print(f"Location: {output_dir}")
        print(f"Total plots: 19 (including separated F1-F3, 3D plots, and Pareto training curves)")


    def plot_best_model_training_results(self, output_dir="outputs/figures", font_size=14):
        """Create figures showing the final accuracy results for the best model.
        
        This method generates publication-ready SVG figures showing:
        1. Training curves (accuracy and loss over epochs) for the best model
        2. Final accuracy summary with model specifications
        3. Performance comparison with other top models
        
        Parameters:
        -----------
        output_dir : str, default="plots"
            Directory to save the plots
        font_size : int, default=14
            Base font size for all text elements
        """
        if self.data is None:
            print("[ERROR] No data loaded. Cannot generate best model plots.")
            return
        
        # Use output_dir directly without adding subfolder (already handled by caller)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find the best model
        best_model_idx = self.data['val_acc'].idxmax()
        best_model = self.data.loc[best_model_idx]
        best_eval_id = best_model['eval_id']
        
        print(f"🏆 BEST MODEL IDENTIFIED:")
        print(f"   Eval ID: {best_eval_id}")
        print(f"   Final Accuracy: {best_model['val_acc']:.2f}%")
        print(f"   Configuration: {best_model['embed']} embedding, {int(best_model['n_qubits'])} qubits, depth {int(best_model['depth'])}")
        print(f"   Circuit Cost (F2): {int(best_model['f2_circuit_cost'])}")
        print(f"   Subcircuits (F3): {int(best_model['f3_n_subcircuits'])}")
        
        # Load training epoch data
        train_log_path = Path(f"{self.csv_path.parent}/train_epoch_log.csv")
        if not train_log_path.exists():
            print(f"[ERROR] Training log file not found: {train_log_path}")
            return
        
        try:
            train_data = pd.read_csv(train_log_path)
            print(f"Loaded training data: {len(train_data)} records")
        except Exception as e:
            print(f"[ERROR] Error loading training data: {e}")
            return
        
        # Filter data for the best model
        best_model_data = train_data[train_data['eval_id'] == best_eval_id].copy()
        if len(best_model_data) == 0:
            print(f"[ERROR] No training data found for best model {best_eval_id}")
            return
        
        # Keep only epoch_end records for clean curves
        epoch_data = best_model_data[best_model_data['phase'] == 'epoch_end'].copy()
        epoch_data = epoch_data.sort_values('epoch')
        
        print(f"📈 Best model training epochs: {len(epoch_data)}")
        
        # Figure 1: Training curves (Accuracy and Loss)
        with sns.axes_style("whitegrid"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Accuracy subplot
            if 'train_acc' in epoch_data.columns and 'val_acc' in epoch_data.columns:
                ax1.plot(epoch_data['epoch'], epoch_data['train_acc'], 'o-', 
                        label='Training Accuracy', linewidth=2, markersize=6, color='#1f77b4')
                ax1.plot(epoch_data['epoch'], epoch_data['val_acc'], 's-', 
                        label='Validation Accuracy', linewidth=2, markersize=6, color='#ff7f0e')
                ax1.set_ylabel('Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
                ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=font_size-1)
            else:
                ax1.text(0.5, 0.5, 'Training accuracy data not available', 
                        ha='center', va='center', transform=ax1.transAxes, 
                        fontsize=font_size, style='italic')
            
            ax1.set_xlabel('Epoch', fontsize=font_size, fontweight='bold', labelpad=15)
            ax1.tick_params(axis='both', which='major', labelsize=font_size-1)
            ax1.grid(True, alpha=0.3)
            
            # Loss subplot
            if 'train_loss' in epoch_data.columns and 'val_loss' in epoch_data.columns:
                ax2.plot(epoch_data['epoch'], epoch_data['train_loss'], 'o-', 
                        label='Training Loss', linewidth=2, markersize=6, color='#d62728')
                ax2.plot(epoch_data['epoch'], epoch_data['val_loss'], 's-', 
                        label='Validation Loss', linewidth=2, markersize=6, color='#9467bd')
                ax2.set_ylabel('Loss', fontsize=font_size, fontweight='bold', labelpad=15)
                ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=font_size-1)
            else:
                ax2.text(0.5, 0.5, 'Training loss data not available', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=font_size, style='italic')
            
            ax2.set_xlabel('Epoch', fontsize=font_size, fontweight='bold', labelpad=15)
            ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/best_model_training_curves.svg", format='svg', bbox_inches='tight')
            plt.show()
        
        # Figure 2: Final accuracy summary with model specs
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create a summary bar chart
            metrics = ['Final Validation Accuracy']
            values = [best_model['val_acc']]
            colors = ['#2ca02c']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8, width=0.6)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{value:.2f}%', ha='center', va='bottom', 
                       fontsize=font_size+2, fontweight='bold')
            
            ax.set_ylabel('Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
            ax.set_ylim(0, max(values) * 1.15)  # Add 15% padding
            ax.tick_params(axis='both', which='major', labelsize=font_size-1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add model specifications as text
            specs_text = f"""Best Model Specifications:
• Embedding: {best_model['embed']}
• Qubits: {int(best_model['n_qubits'])}
• Circuit Depth: {int(best_model['depth'])}
• Circuit Cost (F2): {int(best_model['f2_circuit_cost'])}
• Subcircuits (F3): {int(best_model['f3_n_subcircuits'])}
• Training Time: {best_model['seconds']:.1f}s
• Learning Rate: {best_model['learning_rate']:.2e}"""
            
            ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=font_size-1,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/best_model_final_accuracy.svg", format='svg', bbox_inches='tight')
            plt.show()
        
        # Figure 3: Performance comparison with top 5 models
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Get top 5 models
            top_5 = self.data.nlargest(5, 'val_acc')
            
            # Create horizontal bar chart
            colors = ['#2ca02c' if x == best_eval_id else '#1f77b4' for x in top_5['eval_id']]
            bars = ax.barh(range(len(top_5)), top_5['val_acc'], color=colors, alpha=0.8)
            
            # Add model labels
            labels = []
            for _, model in top_5.iterrows():
                label = f"{model['eval_id']}\n({model['embed']}, {int(model['n_qubits'])}q, d{int(model['depth'])})"
                if model['eval_id'] == best_eval_id:
                    label += " ← BEST"
                labels.append(label)
            
            ax.set_yticks(range(len(top_5)))
            ax.set_yticklabels(labels, fontsize=font_size-2)
            ax.set_xlabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold', labelpad=15)
            ax.tick_params(axis='both', which='major', labelsize=font_size-1)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, acc) in enumerate(zip(bars, top_5['val_acc'])):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{acc:.2f}%', ha='left', va='center', 
                       fontsize=font_size-1, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/best_model_comparison.svg", format='svg', bbox_inches='tight')
            plt.show()
        
        print(f"\n[OK] BEST MODEL TRAINING RESULTS GENERATED:")
        print(f"   Training curves: {output_dir}/best_model_training_curves.svg")
        print(f"   🏆 Final accuracy: {output_dir}/best_model_final_accuracy.svg") 
        print(f"   📈 Performance comparison: {output_dir}/best_model_comparison.svg")
        print(f"   🎯 Best model achieved: {best_model['val_acc']:.2f}% validation accuracy")

    def plot_pareto_training_curves(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Plot training curves for all final pareto optimal points.
        
        This method extracts training data for all final pareto optimal configurations
        and creates comprehensive training curve visualizations.
        
        Parameters:
        -----------
        output_dir : str, default="outputs/figures"
            Directory to save the plots
        font_size : int, default=12
            Base font size for plots
        file_ext : str, default="png"
            File extension for saved plots
        """
        output_path = self.create_output_dir(output_dir)
        
        # Load training epoch data
        train_log_path = Path(f"{self.csv_path.parent}/train_epoch_log.csv")
        if not train_log_path.exists():
            print(f"[ERROR] Training log file not found: {train_log_path}")
            return
        
        try:
            train_data = pd.read_csv(train_log_path)
            print(f"Loaded training data: {len(train_data)} records")
        except Exception as e:
            print(f"[ERROR] Error loading training data: {e}")
            return
        
        # Auto-detect pareto optimal eval_ids
        pareto_ids = train_data[train_data['eval_id'].str.startswith('final-pareto-')]['eval_id'].unique().tolist()
        if len(pareto_ids) == 0:
            print("[WARN] No pareto optimal points found (no eval_ids starting with 'final-pareto-')")
            return
        
        print(f"🔍 Found {len(pareto_ids)} pareto optimal points: {pareto_ids}")
        
        # Extract training data for each pareto point
        pareto_data = {}
        for eval_id in pareto_ids:
            eval_data = train_data[train_data['eval_id'] == eval_id].copy()
            if len(eval_data) == 0:
                print(f"[WARN] Warning: No data found for {eval_id}")
                continue
            
            # Keep only epoch_end records for clean curves
            epoch_data = eval_data[eval_data['phase'] == 'epoch_end'].copy()
            epoch_data = epoch_data.sort_values('epoch')
            
            if len(epoch_data) == 0:
                print(f"[WARN] Warning: No epoch_end data found for {eval_id}")
                continue
            
            pareto_data[eval_id] = epoch_data
            print(f"  {eval_id}: {len(epoch_data)} epochs")
        
        if len(pareto_data) == 0:
            print("[ERROR] No training data found for any pareto optimal points")
            return
        
        # Color palette for different pareto points
        colors = {
            'final-pareto-best_accuracy': '#2ca02c',  # Green
            'final-pareto-lowest_cost': '#d62728',    # Red
            'final-pareto-balanced_2': '#1f77b4',     # Blue
        }
        
        # Default colors if eval_id not in dict
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Training and Validation Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (eval_id, data) in enumerate(pareto_data.items()):
            color = colors.get(eval_id, default_colors[i % len(default_colors)])
            label = eval_id.replace('final-pareto-', '').replace('_', ' ').title()
            
            if 'train_acc' in data.columns and not data['train_acc'].isna().all():
                ax1.plot(data['epoch'], data['train_acc'], 'o--', 
                        label=f'{label} (Train)', linewidth=2, markersize=6, 
                        color=color, alpha=0.7)
            
            if 'val_acc' in data.columns and not data['val_acc'].isna().all():
                ax1.plot(data['epoch'], data['val_acc'], 's-', 
                        label=f'{label} (Val)', linewidth=2, markersize=6, 
                        color=color, alpha=1.0)
        
        ax1.set_xlabel('Epoch', fontsize=font_size, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=font_size, fontweight='bold')
        ax1.set_title('Training and Validation Accuracy', fontsize=font_size+2, fontweight='bold')
        ax1.legend(fontsize=font_size-2, loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=font_size-1)
        
        # Plot 2: Training and Validation Loss
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (eval_id, data) in enumerate(pareto_data.items()):
            color = colors.get(eval_id, default_colors[i % len(default_colors)])
            label = eval_id.replace('final-pareto-', '').replace('_', ' ').title()
            
            if 'train_loss' in data.columns and not data['train_loss'].isna().all():
                ax2.plot(data['epoch'], data['train_loss'], 'o--', 
                        label=f'{label} (Train)', linewidth=2, markersize=6, 
                        color=color, alpha=0.7)
            
            if 'val_loss' in data.columns and not data['val_loss'].isna().all():
                ax2.plot(data['epoch'], data['val_loss'], 's-', 
                        label=f'{label} (Val)', linewidth=2, markersize=6, 
                        color=color, alpha=1.0)
        
        ax2.set_xlabel('Epoch', fontsize=font_size, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=font_size, fontweight='bold')
        ax2.set_title('Training and Validation Loss', fontsize=font_size+2, fontweight='bold')
        ax2.legend(fontsize=font_size-2, loc='best', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
        
        # Plot 3: Validation Accuracy Comparison (overlay)
        ax3 = fig.add_subplot(gs[1, 0])
        for i, (eval_id, data) in enumerate(pareto_data.items()):
            color = colors.get(eval_id, default_colors[i % len(default_colors)])
            label = eval_id.replace('final-pareto-', '').replace('_', ' ').title()
            
            if 'val_acc' in data.columns and not data['val_acc'].isna().all():
                ax3.plot(data['epoch'], data['val_acc'], 'o-', 
                        label=label, linewidth=3, markersize=8, 
                        color=color, alpha=0.8)
        
        ax3.set_xlabel('Epoch', fontsize=font_size, fontweight='bold')
        ax3.set_ylabel('Validation Accuracy (%)', fontsize=font_size, fontweight='bold')
        ax3.set_title('Validation Accuracy Comparison', fontsize=font_size+2, fontweight='bold')
        ax3.legend(fontsize=font_size-2, loc='best', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=font_size-1)
        
        # Plot 4: Final Performance Summary
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Extract final epoch data
        final_data = []
        for eval_id, data in pareto_data.items():
            if len(data) > 0:
                final_row = data.iloc[-1]
                label = eval_id.replace('final-pareto-', '').replace('_', ' ').title()
                final_data.append({
                    'label': label,
                    'val_acc': final_row.get('val_acc', 0),
                    'train_acc': final_row.get('train_acc', 0),
                    'val_loss': final_row.get('val_loss', 0),
                    'train_loss': final_row.get('train_loss', 0),
                    'eval_id': eval_id
                })
        
        if final_data:
            final_df = pd.DataFrame(final_data)
            x_pos = range(len(final_df))
            
            # Create grouped bar chart
            width = 0.35
            ax4.bar([x - width/2 for x in x_pos], final_df['train_acc'], 
                   width, label='Train Acc', alpha=0.7, color='#1f77b4')
            ax4.bar([x + width/2 for x in x_pos], final_df['val_acc'], 
                   width, label='Val Acc', alpha=0.7, color='#ff7f0e')
            
            ax4.set_xlabel('Pareto Optimal Point', fontsize=font_size, fontweight='bold')
            ax4.set_ylabel('Accuracy (%)', fontsize=font_size, fontweight='bold')
            ax4.set_title('Final Epoch Performance', fontsize=font_size+2, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(final_df['label'], rotation=15, ha='right', fontsize=font_size-2)
            ax4.legend(fontsize=font_size-2, frameon=True, fancybox=True, shadow=True)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.tick_params(axis='both', which='major', labelsize=font_size-1)
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(final_df.iterrows()):
                ax4.text(i - width/2, row['train_acc'] + 0.5, f"{row['train_acc']:.1f}%",
                        ha='center', va='bottom', fontsize=font_size-3, fontweight='bold')
                ax4.text(i + width/2, row['val_acc'] + 0.5, f"{row['val_acc']:.1f}%",
                        ha='center', va='bottom', fontsize=font_size-3, fontweight='bold')
        
        plt.suptitle('Pareto Optimal Points - Training Curves', 
                    fontsize=font_size+4, fontweight='bold', y=0.995)
        
        # Save figure
        filename = f"{output_path}/pareto_training_curves.{file_ext}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to: {filename}")
        plt.close()

    def plot_epoch_bars(self, run_dir=None, output_dir="outputs/figures", font_size=12):
        """Plot bar chart showing 3 epochs for each candidate with final best highlighted.
        
        This function creates a grouped bar chart where each candidate has 3 bars
        representing their accuracy at epochs 1, 2, and 3. The final best candidate
        is highlighted in a different color (gold) for easy identification.
        
        Parameters:
        -----------
        run_dir : str or Path, optional
            Path to the run directory containing epoch_correlation.csv.
            If None, will look in self.log_folder/run_*/ directories.
        output_dir : str, default="plots"
            Directory to save the plot
        font_size : int, default=12
            Base font size for text elements
        """
        import pandas as pd
        from pathlib import Path
        import numpy as np
        
        # Determine the epoch_correlation.csv path
        if run_dir is None:
            if self.log_folder:
                # Find run directories in the log folder
                log_path = Path(f"logs/{self.log_folder}")
                run_dirs = list(log_path.glob("run_*"))
                if not run_dirs:
                    print(f"[ERROR] No run directories found in {log_path}")
                    return
                run_dir = run_dirs[0]  # Use the first run directory
                print(f"[INFO] Using run directory: {run_dir}")
            else:
                print("[ERROR] No run_dir specified and no log_folder set")
                return
        else:
            # Convert string to Path and make it relative to log_folder if needed
            run_dir = Path(run_dir)
            if not run_dir.is_absolute() and not run_dir.exists():
                # Try to find it in the log folder
                if self.log_folder:
                    run_dir = Path(f"logs/{self.log_folder}") / run_dir.name
                else:
                    run_dir = Path("logs") / run_dir.name
            print(f"[INFO] Using run directory: {run_dir}")
        
        epoch_csv_path = run_dir / "epoch_correlation.csv"
        
        if not epoch_csv_path.exists():
            print(f"[ERROR] File not found: {epoch_csv_path}")
            print(f"[INFO] Please run: python correlate_nsga_vs_final.py <log_dir> --analyze-run <run_name>")
            return
        
        # Load data
        try:
            df = pd.read_csv(epoch_csv_path)
            print(f"[INFO] Loaded {len(df)} rows from {epoch_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load {epoch_csv_path}: {e}")
            return
        
        # Create output directory (output_dir already includes subfolder if needed)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get unique candidates and sort by label for better organization
        candidates = df.groupby('eval_id').first().reset_index()
        candidates = candidates.sort_values(['label', 'eval_id'])
        
        # Identify final best
        final_best_id = df[df['label'] == 'final_best']['eval_id'].iloc[0] if 'final_best' in df['label'].values else None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Set up bar positions - now with 4 bars per candidate
        n_candidates = len(candidates)
        bar_width = 0.2
        x_positions = np.arange(n_candidates)
        
        # Prepare data for each epoch
        pre_val_acc_list = []  # Store pre-validation accuracy
        epoch_1_acc = []
        epoch_2_acc = []
        epoch_3_acc = []
        labels = []
        colors_pre = []
        colors_e1 = []
        colors_e2 = []
        colors_e3 = []
        
        for idx, row in candidates.iterrows():
            eval_id = row['eval_id']
            candidate_data = df[df['eval_id'] == eval_id].sort_values('epoch')
            
            # Get accuracy for each epoch (handle missing epochs)
            epochs = candidate_data['epoch'].values
            accs = candidate_data['val_acc'].values
            pre_accs = candidate_data['pre_val_acc'].values
            
            e1 = accs[epochs == 1][0] if 1 in epochs else 0
            e2 = accs[epochs == 2][0] if 2 in epochs else 0
            e3 = accs[epochs == 3][0] if 3 in epochs else 0
            pre_acc = pre_accs[0] if len(pre_accs) > 0 else 0  # Pre-val accuracy (same for all epochs)
            
            pre_val_acc_list.append(pre_acc)
            epoch_1_acc.append(e1)
            epoch_2_acc.append(e2)
            epoch_3_acc.append(e3)
            
            # Create label
            label_text = row['label']
            labels.append(label_text)
            
            # Set colors - gold for final best, blue for best gen, red for random gen
            is_final = (eval_id == final_best_id)
            if is_final:
                colors_pre.append('#808080')  # Gray for pre-val
                colors_e1.append('#FFD700')  # Gold
                colors_e2.append('#FFA500')  # Orange
                colors_e3.append('#FF8C00')  # Dark Orange
            elif 'best_gen' in label_text:
                colors_pre.append('#B0C4DE')  # Light Steel Blue for pre-val
                colors_e1.append('#6495ED')  # Cornflower Blue
                colors_e2.append('#4169E1')  # Royal Blue
                colors_e3.append('#0000CD')  # Medium Blue
            else:  # random_gen
                colors_pre.append('#FFA07A')  # Light Salmon for pre-val
                colors_e1.append('#FF6B6B')  # Light Red
                colors_e2.append('#EE4B2B')  # Red
                colors_e3.append('#C41E3A')  # Dark Red
        
        # Create bars for each epoch - now with pre-val as the first bar
        bars_pre = ax.bar(x_positions - 1.5*bar_width, pre_val_acc_list, bar_width, 
                         label='Pre-Val', color=colors_pre, alpha=0.7, edgecolor='black', linewidth=0.5, hatch='//')
        bars1 = ax.bar(x_positions - 0.5*bar_width, epoch_1_acc, bar_width, 
                      label='Epoch 1', color=colors_e1, alpha=0.9, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x_positions + 0.5*bar_width, epoch_2_acc, bar_width, 
                      label='Epoch 2', color=colors_e2, alpha=0.9, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x_positions + 1.5*bar_width, epoch_3_acc, bar_width, 
                      label='Epoch 3', color=colors_e3, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars with more offset to prevent overlap
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                if value > 0:  # Only label non-zero bars
                    height = bar.get_height()
                    # Add more offset to prevent overlap with bars
                    ax.text(bar.get_x() + bar.get_width()/2., height + 4,
                           f'{value:.1f}%',
                           ha='center', va='bottom', fontsize=font_size-4, rotation=90)
        
        add_value_labels(bars_pre, pre_val_acc_list)
        add_value_labels(bars1, epoch_1_acc)
        add_value_labels(bars2, epoch_2_acc)
        add_value_labels(bars3, epoch_3_acc)
        
        # Customize plot
        ax.set_xlabel('Candidates', fontsize=font_size+4, fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontsize=font_size+4, fontweight='bold')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size-2)
        ax.tick_params(axis='y', labelsize=font_size)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=font_size+2, framealpha=0.9)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Highlight final best with a box
        if final_best_id:
            final_best_idx = list(candidates['eval_id']).index(final_best_id)
            # Add a rectangle to highlight final best
            from matplotlib.patches import Rectangle
            rect = Rectangle((final_best_idx - 0.5, ax.get_ylim()[0]), 1, ax.get_ylim()[1] - ax.get_ylim()[0],
                           linewidth=3, edgecolor='gold', facecolor='none', linestyle='--', zorder=0)
            ax.add_patch(rect)
            
            # Add annotation much higher above the plot area to prevent overlap
            ax.annotate('FINAL BEST', xy=(final_best_idx, ax.get_ylim()[1]),
                       xytext=(final_best_idx, ax.get_ylim()[1] * 1.20),
                       fontsize=font_size+2, fontweight='bold', color='darkgoldenrod',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7),
                       annotation_clip=False)
        
        # Set y-axis limits with extra padding for labels - increased for more space
        all_values = pre_val_acc_list + epoch_1_acc + epoch_2_acc + epoch_3_acc
        ax.set_ylim(0, max(all_values) * 1.5)
        
        plt.tight_layout()
        
        # Save figure
        output_file = f"{output_dir}/epoch_bars_comparison.svg"
        plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=300)
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("EPOCH BAR CHART SUMMARY")
        print("="*70)
        print(f"\n📈 Total candidates plotted: {n_candidates}")
        print(f"   - 4 bars per candidate (Pre-Val + Epochs 1, 2, 3)")
        print(f"   - Total bars: {n_candidates * 4}")
        
        if final_best_id:
            fb_data = df[df['eval_id'] == final_best_id].sort_values('epoch')
            fb_pre = fb_data.iloc[0]['pre_val_acc']
            print(f"\n🏆 Final Best Candidate: {final_best_id}")
            print(f"   - Highlighted in GOLD/ORANGE colors")
            print(f"   - Pre-Val: {fb_pre:.2f}%")
            print(f"   - Epoch 1: {fb_data.iloc[0]['val_acc']:.2f}%")
            print(f"   - Epoch 2: {fb_data.iloc[1]['val_acc']:.2f}%")
            print(f"   - Epoch 3: {fb_data.iloc[2]['val_acc']:.2f}%")
            print(f"   - Improvement from Pre-Val: {fb_data.iloc[2]['val_acc'] - fb_pre:.2f}%")
        
        print(f"\n[OK] Bar chart saved to: {output_file}")
        print("="*70)

    def plot_search_space_exploration(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Visualize the search space exploration over generations."""
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Qubits vs Depth exploration heatmap
            qubit_depth_counts = self.data.groupby(['n_qubits', 'depth']).size().unstack(fill_value=0)
            im1 = axes[0, 0].imshow(qubit_depth_counts.values, cmap='YlOrRd', aspect='auto', origin='lower')
            axes[0, 0].set_xticks(range(len(qubit_depth_counts.columns)))
            axes[0, 0].set_xticklabels(qubit_depth_counts.columns)
            axes[0, 0].set_yticks(range(len(qubit_depth_counts.index)))
            axes[0, 0].set_yticklabels(qubit_depth_counts.index)
            axes[0, 0].set_xlabel('Circuit Depth', fontsize=font_size, fontweight='bold')
            axes[0, 0].set_ylabel('Number of Qubits', fontsize=font_size, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0], label='Number of Evaluations')
            
            # Add text annotations for counts
            for i in range(len(qubit_depth_counts.index)):
                for j in range(len(qubit_depth_counts.columns)):
                    count = qubit_depth_counts.iloc[i, j]
                    if count > 0:
                        axes[0, 0].text(j, i, str(int(count)), ha='center', va='center', 
                                      fontsize=font_size-2, fontweight='bold', color='white' if count > qubit_depth_counts.values.max()/2 else 'black')
            
            # 2. Embedding type distribution over generations
            embed_gen_counts = self.data.groupby(['generation', 'embed']).size().unstack(fill_value=0)
            embed_gen_counts.plot(kind='bar', stacked=True, ax=axes[0, 1], colormap='Set2', width=0.8)
            axes[0, 1].set_xlabel('Generation', fontsize=font_size, fontweight='bold')
            axes[0, 1].set_ylabel('Number of Evaluations', fontsize=font_size, fontweight='bold')
            axes[0, 1].legend(title='Embedding Type', fontsize=font_size-2, title_fontsize=font_size-1)
            axes[0, 1].tick_params(axis='x', rotation=0, labelsize=font_size-2)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 3. Learning rate distribution exploration
            lr_bins = pd.cut(self.data['learning_rate'], bins=15)
            lr_dist = lr_bins.value_counts().sort_index()
            axes[1, 0].bar(range(len(lr_dist)), lr_dist.values, alpha=0.7, color='steelblue', edgecolor='black')
            axes[1, 0].set_xlabel('Learning Rate Bins', fontsize=font_size, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontsize=font_size, fontweight='bold')
            axes[1, 0].set_xticks(range(0, len(lr_dist), max(1, len(lr_dist)//5)))
            axes[1, 0].set_xticklabels([str(lr_dist.index[i])[:20] for i in range(0, len(lr_dist), max(1, len(lr_dist)//5))], 
                                      rotation=45, ha='right', fontsize=font_size-3)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 4. Search space coverage over generations (cumulative unique configurations)
            unique_configs_per_gen = []
            cumulative_configs = set()
            for gen in sorted(self.data['generation'].unique()):
                gen_data = self.data[self.data['generation'] == gen]
                configs = gen_data[['n_qubits', 'depth', 'embed']].drop_duplicates()
                cumulative_configs.update([tuple(row) for _, row in configs.iterrows()])
                unique_configs_per_gen.append(len(cumulative_configs))
            
            generations = sorted(self.data['generation'].unique())
            axes[1, 1].plot(generations, unique_configs_per_gen, 'o-', linewidth=2, markersize=8, color='green')
            axes[1, 1].fill_between(generations, unique_configs_per_gen, alpha=0.3, color='green')
            axes[1, 1].set_xlabel('Generation', fontsize=font_size, fontweight='bold')
            axes[1, 1].set_ylabel('Cumulative Unique Configurations', fontsize=font_size, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='both', which='major', labelsize=font_size-1)
            
            # Add annotation for total unique configs
            total_unique = len(cumulative_configs)
            axes[1, 1].axhline(total_unique, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                             label=f'Total: {total_unique} unique configs')
            axes[1, 1].legend(fontsize=font_size-1)
            
            plt.tight_layout()
            filename = f"{output_dir}/search_space_exploration.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"[OK] Search space exploration plot saved to: {filename}")

    def plot_parameter_space_coverage(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Visualize parameter space coverage and density - saves individual figures."""
        with sns.axes_style("whitegrid"):
            # 1. 2D parameter space: Qubits vs Depth with performance coloring
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            scatter1 = ax1.scatter(self.data['n_qubits'], self.data['depth'], 
                                 c=self.data['val_acc'], cmap='RdYlGn', 
                                 s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax1.set_xlabel('Number of Qubits', fontsize=font_size+2, fontweight='bold')
            ax1.set_ylabel('Circuit Depth', fontsize=font_size+2, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Validation Accuracy (%)', fontsize=font_size+1)
            cbar1.ax.tick_params(labelsize=font_size)
            plt.tight_layout()
            filename1 = f"{output_dir}/parameter_space_qubits_depth.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename1, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Parameter space exploration by generation
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            for gen in sorted(self.data['generation'].unique())[:5]:  # Show first 5 generations
                gen_data = self.data[self.data['generation'] == gen]
                ax2.scatter(gen_data['n_qubits'], gen_data['depth'], 
                          alpha=0.6, s=60, label=f'Gen {int(gen)}', edgecolors='black', linewidth=0.3)
            ax2.set_xlabel('Number of Qubits', fontsize=font_size+2, fontweight='bold')
            ax2.set_ylabel('Circuit Depth', fontsize=font_size+2, fontweight='bold')
            ax2.legend(fontsize=font_size, ncol=2, frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            filename2 = f"{output_dir}/parameter_space_generations.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename2, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # 3. CNOT mode exploration
            if 'cnot_modes' in self.data.columns:
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                evals_copy = self.data.copy()
                evals_copy['cnot_category'] = evals_copy['cnot_modes'].apply(categorize_cnot_mode)
                cnot_counts = evals_copy['cnot_category'].value_counts()
                ax3.pie(cnot_counts.values, labels=cnot_counts.index, autopct='%1.1f%%',
                       textprops={'fontsize': font_size+1, 'fontweight': 'bold'},
                       colors=plt.cm.Set3(range(len(cnot_counts))))
                plt.tight_layout()
                filename3 = f"{output_dir}/parameter_space_cnot_modes.{file_ext}"
                if file_ext == "svg":
                    plt.savefig(filename3, format='svg', bbox_inches='tight')
                else:
                    plt.savefig(filename3, dpi=300, bbox_inches='tight')
                plt.close(fig3)
            
            # 4. Multi-dimensional parameter space summary
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            ax4.axis('off')
            
            # Calculate statistics
            total_configs = len(self.data[['n_qubits', 'depth', 'embed']].drop_duplicates())
            unique_qubits = self.data['n_qubits'].nunique()
            unique_depths = self.data['depth'].nunique()
            unique_embeds = self.data['embed'].nunique()
            unique_lrs = self.data['learning_rate'].nunique()
            
            stats_text = f"""
            SEARCH SPACE STATISTICS
            
            Total Evaluations: {len(self.data)}
            Unique Configurations: {total_configs}
            
            PARAMETER DIVERSITY:
            • Qubits: {unique_qubits} unique values ({self.data['n_qubits'].min():.0f} - {self.data['n_qubits'].max():.0f})
            • Depths: {unique_depths} unique values ({self.data['depth'].min():.0f} - {self.data['depth'].max():.0f})
            • Embeddings: {unique_embeds} types ({', '.join(self.data['embed'].unique())})
            • Learning Rates: {unique_lrs} unique values
            
            EXPLORATION COVERAGE:
            • Qubit×Depth combinations: {len(self.data[['n_qubits', 'depth']].drop_duplicates())}
            • Full config space: {unique_qubits} × {unique_depths} × {unique_embeds} = {unique_qubits * unique_depths * unique_embeds} possible
            • Coverage: {total_configs / (unique_qubits * unique_depths * unique_embeds) * 100:.1f}%
            
            GENERATION SPAN:
            • Generations: {self.data['generation'].min():.0f} - {self.data['generation'].max():.0f}
            • Avg evaluations/gen: {len(self.data) / self.data['generation'].nunique():.1f}
            """
            
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                    fontsize=font_size+1, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.tight_layout()
            filename4 = f"{output_dir}/parameter_space_statistics.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename4, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename4, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            print(f"[OK] Parameter space coverage plots saved:")
            print(f"   - {filename1}")
            print(f"   - {filename2}")
            if 'cnot_modes' in self.data.columns:
                print(f"   - {filename3}")
            print(f"   - {filename4}")

    def plot_hyperparameter_combinations(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Visualize hyperparameter combinations and their performance - saves individual figures."""
        with sns.axes_style("whitegrid"):
            # 1. Embedding × Qubits heatmap (average accuracy)
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            embed_qubit_acc = self.data.groupby(['embed', 'n_qubits'])['val_acc'].mean().unstack(fill_value=0)
            im1 = ax1.imshow(embed_qubit_acc.values, cmap='RdYlGn', aspect='auto', origin='lower', 
                           vmin=self.data['val_acc'].min(), vmax=self.data['val_acc'].max())
            ax1.set_xticks(range(len(embed_qubit_acc.columns)))
            ax1.set_xticklabels(embed_qubit_acc.columns, fontsize=font_size)
            ax1.set_yticks(range(len(embed_qubit_acc.index)))
            ax1.set_yticklabels(embed_qubit_acc.index, fontsize=font_size)
            ax1.set_xlabel('Number of Qubits', fontsize=font_size+2, fontweight='bold')
            ax1.set_ylabel('Embedding Type', fontsize=font_size+2, fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Accuracy (%)', fontsize=font_size+1)
            cbar1.ax.tick_params(labelsize=font_size)
            plt.tight_layout()
            filename1 = f"{output_dir}/hyperparameter_embedding_qubits.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename1, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Embedding × Depth heatmap
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            embed_depth_acc = self.data.groupby(['embed', 'depth'])['val_acc'].mean().unstack(fill_value=0)
            im2 = ax2.imshow(embed_depth_acc.values, cmap='RdYlGn', aspect='auto', origin='lower', 
                           vmin=self.data['val_acc'].min(), vmax=self.data['val_acc'].max())
            ax2.set_xticks(range(len(embed_depth_acc.columns)))
            ax2.set_xticklabels(embed_depth_acc.columns, fontsize=font_size)
            ax2.set_yticks(range(len(embed_depth_acc.index)))
            ax2.set_yticklabels(embed_depth_acc.index, fontsize=font_size)
            ax2.set_xlabel('Circuit Depth', fontsize=font_size+2, fontweight='bold')
            ax2.set_ylabel('Embedding Type', fontsize=font_size+2, fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Accuracy (%)', fontsize=font_size+1)
            cbar2.ax.tick_params(labelsize=font_size)
            plt.tight_layout()
            filename2 = f"{output_dir}/hyperparameter_embedding_depth.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename2, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # 3. Qubits × Depth heatmap (count of evaluations)
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            qubit_depth_counts = self.data.groupby(['n_qubits', 'depth']).size().unstack(fill_value=0)
            im3 = ax3.imshow(qubit_depth_counts.values, cmap='YlOrRd', aspect='auto', origin='lower')
            ax3.set_xticks(range(len(qubit_depth_counts.columns)))
            ax3.set_xticklabels(qubit_depth_counts.columns, fontsize=font_size)
            ax3.set_yticks(range(len(qubit_depth_counts.index)))
            ax3.set_yticklabels(qubit_depth_counts.index, fontsize=font_size)
            ax3.set_xlabel('Circuit Depth', fontsize=font_size+2, fontweight='bold')
            ax3.set_ylabel('Number of Qubits', fontsize=font_size+2, fontweight='bold')
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('Count', fontsize=font_size+1)
            cbar3.ax.tick_params(labelsize=font_size)
            plt.tight_layout()
            filename3 = f"{output_dir}/hyperparameter_qubits_depth_count.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename3, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename3, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # 4. Best configuration per embedding type
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            best_per_embed = self.data.loc[self.data.groupby('embed')['val_acc'].idxmax()]
            bars = ax4.bar(range(len(best_per_embed)), best_per_embed['val_acc'], 
                          color=plt.cm.Set2(range(len(best_per_embed))), alpha=0.8, edgecolor='black', linewidth=1.5)
            ax4.set_xticks(range(len(best_per_embed)))
            ax4.set_xticklabels(best_per_embed['embed'], rotation=45, ha='right', fontsize=font_size+1)
            ax4.set_ylabel('Best Accuracy (%)', fontsize=font_size+2, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            for i, (bar, acc) in enumerate(zip(bars, best_per_embed['val_acc'])):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=font_size+1, fontweight='bold')
            plt.tight_layout()
            filename4 = f"{output_dir}/hyperparameter_best_per_embedding.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename4, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename4, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # 5. Configuration frequency vs performance
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            config_counts = self.data.groupby(['n_qubits', 'depth', 'embed']).size().reset_index(name='count')
            config_perf = self.data.groupby(['n_qubits', 'depth', 'embed'])['val_acc'].mean().reset_index(name='avg_acc')
            config_stats = pd.merge(config_counts, config_perf, on=['n_qubits', 'depth', 'embed'])
            scatter5 = ax5.scatter(config_stats['count'], config_stats['avg_acc'], 
                                 c=config_stats['n_qubits'], cmap='viridis', 
                                 s=120, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax5.set_xlabel('Number of Evaluations', fontsize=font_size+2, fontweight='bold')
            ax5.set_ylabel('Average Accuracy (%)', fontsize=font_size+2, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            cbar5 = plt.colorbar(scatter5, ax=ax5)
            cbar5.set_label('Number of Qubits', fontsize=font_size+1)
            cbar5.ax.tick_params(labelsize=font_size)
            plt.tight_layout()
            filename5 = f"{output_dir}/hyperparameter_frequency_performance.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename5, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename5, dpi=300, bbox_inches='tight')
            plt.close(fig5)
            
            # 6. Learning rate vs performance by embedding
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            for embed in self.data['embed'].unique():
                embed_data = self.data[self.data['embed'] == embed]
                ax6.scatter(embed_data['learning_rate'], embed_data['val_acc'], 
                          alpha=0.7, s=60, label=embed, edgecolors='black', linewidth=0.3)
            ax6.set_xlabel('Learning Rate', fontsize=font_size+2, fontweight='bold')
            ax6.set_ylabel('Validation Accuracy (%)', fontsize=font_size+2, fontweight='bold')
            ax6.legend(fontsize=font_size, frameon=True, fancybox=True, shadow=True)
            ax6.grid(True, alpha=0.3)
            ax6.set_xscale('log')
            plt.tight_layout()
            filename6 = f"{output_dir}/hyperparameter_lr_performance.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename6, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename6, dpi=300, bbox_inches='tight')
            plt.close(fig6)
            
            # 7. Parameter space exploration timeline
            fig7, ax7 = plt.subplots(figsize=(14, 8))
            gen_stats = self.data.groupby('generation').agg({
                'n_qubits': ['min', 'max', 'nunique'],
                'depth': ['min', 'max', 'nunique'],
                'val_acc': 'mean'
            })
            generations = gen_stats.index
            ax7_twin = ax7.twinx()
            line1 = ax7.plot(generations, gen_stats[('n_qubits', 'nunique')], 'o-', 
                           label='Unique Qubits', linewidth=3, markersize=8, color='blue')
            line2 = ax7.plot(generations, gen_stats[('depth', 'nunique')], 's-', 
                           label='Unique Depths', linewidth=3, markersize=8, color='green')
            line3 = ax7_twin.plot(generations, gen_stats[('val_acc', 'mean')], '^-', 
                                 label='Mean Accuracy', linewidth=3, markersize=8, color='red')
            ax7.set_xlabel('Generation', fontsize=font_size+2, fontweight='bold')
            ax7.set_ylabel('Number of Unique Values', fontsize=font_size+2, fontweight='bold', color='black')
            ax7_twin.set_ylabel('Mean Accuracy (%)', fontsize=font_size+2, fontweight='bold', color='red')
            ax7.tick_params(axis='y', labelcolor='black', labelsize=font_size)
            ax7_twin.tick_params(axis='y', labelcolor='red', labelsize=font_size)
            ax7.tick_params(axis='x', labelsize=font_size)
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax7.legend(lines, labels, loc='upper left', fontsize=font_size, frameon=True, fancybox=True, shadow=True)
            ax7.grid(True, alpha=0.3)
            plt.tight_layout()
            filename7 = f"{output_dir}/hyperparameter_diversity_timeline.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename7, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename7, dpi=300, bbox_inches='tight')
            plt.close(fig7)
            
            print(f"[OK] Hyperparameter combinations plots saved:")
            print(f"   - {filename1}")
            print(f"   - {filename2}")
            print(f"   - {filename3}")
            print(f"   - {filename4}")
            print(f"   - {filename5}")
            print(f"   - {filename6}")
            print(f"   - {filename7}")

    def plot_parameter_ranges_explored(self, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Visualize parameter ranges and explored search space - saves individual figures."""
        with sns.axes_style("whitegrid"):
            # Calculate parameter statistics
            param_stats = {
                'n_qubits': {
                    'min': self.data['n_qubits'].min(),
                    'max': self.data['n_qubits'].max(),
                    'unique': sorted(self.data['n_qubits'].unique().tolist()),
                    'counts': self.data['n_qubits'].value_counts().sort_index()
                },
                'depth': {
                    'min': self.data['depth'].min(),
                    'max': self.data['depth'].max(),
                    'unique': sorted(self.data['depth'].unique().tolist()),
                    'counts': self.data['depth'].value_counts().sort_index()
                },
                'learning_rate': {
                    'min': self.data['learning_rate'].min(),
                    'max': self.data['learning_rate'].max(),
                    'unique_count': self.data['learning_rate'].nunique()
                },
                'embed': {
                    'unique': self.data['embed'].unique().tolist(),
                    'counts': self.data['embed'].value_counts()
                }
            }
            
            # Calculate coverage statistics
            total_possible_configs = (len(param_stats['n_qubits']['unique']) * 
                                     len(param_stats['depth']['unique']) * 
                                     len(param_stats['embed']['unique']))
            explored_configs = len(self.data[['n_qubits', 'depth', 'embed']].drop_duplicates())
            coverage_pct = (explored_configs / total_possible_configs * 100) if total_possible_configs > 0 else 0
            
            # 1. Parameter ranges summary (text)
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            ax1.axis('off')
            ranges_text = f"""
            PARAMETER RANGES EXPLORED
            
            Number of Qubits:
            • Range: {param_stats['n_qubits']['min']:.0f} - {param_stats['n_qubits']['max']:.0f}
            • Values: {', '.join(map(str, param_stats['n_qubits']['unique']))}
            • Total explored: {len(param_stats['n_qubits']['unique'])} unique values
            
            Circuit Depth:
            • Range: {param_stats['depth']['min']:.0f} - {param_stats['depth']['max']:.0f}
            • Values: {', '.join(map(str, param_stats['depth']['unique']))}
            • Total explored: {len(param_stats['depth']['unique'])} unique values
            
            Learning Rate:
            • Range: {param_stats['learning_rate']['min']:.2e} - {param_stats['learning_rate']['max']:.2e}
            • Unique values: {param_stats['learning_rate']['unique_count']}
            
            Embedding Types:
            • Types: {', '.join(param_stats['embed']['unique'])}
            • Total: {len(param_stats['embed']['unique'])} types
            
            Total Configurations:
            • Unique combinations: {len(self.data[['n_qubits', 'depth', 'embed']].drop_duplicates())}
            • Total evaluations: {len(self.data)}
            """
            ax1.text(0.1, 0.5, ranges_text, transform=ax1.transAxes, 
                    fontsize=font_size+1, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.tight_layout()
            filename1 = f"{output_dir}/parameter_ranges_summary.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename1, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Qubits distribution
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            bars2 = ax2.bar(param_stats['n_qubits']['counts'].index,
                           param_stats['n_qubits']['counts'].values,
                           alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('Number of Qubits', fontsize=font_size+2, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=font_size+2, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=font_size+1, fontweight='bold')
            plt.tight_layout()
            filename2 = f"{output_dir}/parameter_ranges_qubits_dist.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename2, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # 3. Depth distribution
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            bars3 = ax3.bar(param_stats['depth']['counts'].index,
                           param_stats['depth']['counts'].values,
                           alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
            ax3.set_xlabel('Circuit Depth', fontsize=font_size+2, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=font_size+2, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=font_size+1, fontweight='bold')
            plt.tight_layout()
            filename3 = f"{output_dir}/parameter_ranges_depth_dist.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename3, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename3, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # 4. Learning rate distribution (log scale)
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            ax4.hist(self.data['learning_rate'], bins=30, alpha=0.8, color='green', edgecolor='black', linewidth=1.5)
            ax4.set_xlabel('Learning Rate', fontsize=font_size+2, fontweight='bold')
            ax4.set_ylabel('Frequency', fontsize=font_size+2, fontweight='bold')
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axvline(param_stats['learning_rate']['min'], color='red', linestyle='--', 
                        linewidth=3, label=f"Min: {param_stats['learning_rate']['min']:.2e}")
            ax4.axvline(param_stats['learning_rate']['max'], color='red', linestyle='--', 
                        linewidth=3, label=f"Max: {param_stats['learning_rate']['max']:.2e}")
            ax4.legend(fontsize=font_size, frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()
            filename4 = f"{output_dir}/parameter_ranges_lr_dist.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename4, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename4, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # 5. Embedding type distribution
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            bars5 = ax5.bar(range(len(param_stats['embed']['counts'])),
                           param_stats['embed']['counts'].values,
                           color=plt.cm.Set2(range(len(param_stats['embed']['counts']))),
                           alpha=0.8, edgecolor='black', linewidth=1.5)
            ax5.set_xticks(range(len(param_stats['embed']['counts'])))
            ax5.set_xticklabels(param_stats['embed']['counts'].index, rotation=45, ha='right', fontsize=font_size+1)
            ax5.set_ylabel('Frequency', fontsize=font_size+2, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            for i, (bar, count) in enumerate(zip(bars5, param_stats['embed']['counts'].values)):
                ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{int(count)}', ha='center', va='bottom', fontsize=font_size+1, fontweight='bold')
            plt.tight_layout()
            filename5 = f"{output_dir}/parameter_ranges_embedding_dist.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename5, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename5, dpi=300, bbox_inches='tight')
            plt.close(fig5)
            
            # 6. Explored parameter space: Qubits × Depth heatmap
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            qubit_depth_matrix = self.data.groupby(['n_qubits', 'depth']).size().unstack(fill_value=0)
            im6 = ax6.imshow(qubit_depth_matrix.values, cmap='YlOrRd', aspect='auto', origin='lower')
            ax6.set_xticks(range(len(qubit_depth_matrix.columns)))
            ax6.set_xticklabels(qubit_depth_matrix.columns, fontsize=font_size)
            ax6.set_yticks(range(len(qubit_depth_matrix.index)))
            ax6.set_yticklabels(qubit_depth_matrix.index, fontsize=font_size)
            ax6.set_xlabel('Circuit Depth', fontsize=font_size+2, fontweight='bold')
            ax6.set_ylabel('Number of Qubits', fontsize=font_size+2, fontweight='bold')
            cbar6 = plt.colorbar(im6, ax=ax6)
            cbar6.set_label('Evaluations', fontsize=font_size+1)
            cbar6.ax.tick_params(labelsize=font_size)
            # Add count annotations
            for i in range(len(qubit_depth_matrix.index)):
                for j in range(len(qubit_depth_matrix.columns)):
                    count = qubit_depth_matrix.iloc[i, j]
                    if count > 0:
                        ax6.text(j, i, str(int(count)), ha='center', va='center',
                               fontsize=font_size, fontweight='bold',
                               color='white' if count > qubit_depth_matrix.values.max()/2 else 'black')
            plt.tight_layout()
            filename6 = f"{output_dir}/parameter_ranges_qubits_depth_heatmap.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename6, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename6, dpi=300, bbox_inches='tight')
            plt.close(fig6)
            
            # 7. Parameter ranges visualization (parallel coordinates style)
            fig7, ax7 = plt.subplots(figsize=(12, 8))
            # Normalize parameters to 0-1 range for visualization
            n_qubits_norm = (self.data['n_qubits'] - param_stats['n_qubits']['min']) / \
                           (param_stats['n_qubits']['max'] - param_stats['n_qubits']['min'] + 1e-10)
            depth_norm = (self.data['depth'] - param_stats['depth']['min']) / \
                        (param_stats['depth']['max'] - param_stats['depth']['min'] + 1e-10)
            lr_norm = (np.log10(self.data['learning_rate']) - np.log10(param_stats['learning_rate']['min'])) / \
                     (np.log10(param_stats['learning_rate']['max']) - np.log10(param_stats['learning_rate']['min']) + 1e-10)
            # Create parallel coordinates plot
            y_positions = [0, 1, 2]
            param_names = ['Qubits', 'Depth', 'Learning Rate']
            # Plot lines for each configuration
            for idx in range(min(100, len(self.data))):  # Limit to 100 for clarity
                values = [n_qubits_norm.iloc[idx], depth_norm.iloc[idx], lr_norm.iloc[idx]]
                ax7.plot(y_positions, values, alpha=0.3, linewidth=0.5, color='gray')
            # Plot mean lines
            mean_values = [n_qubits_norm.mean(), depth_norm.mean(), lr_norm.mean()]
            ax7.plot(y_positions, mean_values, 'o-', linewidth=3, color='red', 
                    markersize=12, label='Mean', zorder=10)
            # Add min/max markers
            min_values = [0, 0, 0]
            max_values = [1, 1, 1]
            ax7.plot(y_positions, min_values, 's', markersize=10, color='blue', 
                    label='Min', zorder=10)
            ax7.plot(y_positions, max_values, '^', markersize=10, color='green', 
                    label='Max', zorder=10)
            ax7.set_xticks(y_positions)
            ax7.set_xticklabels(param_names, fontsize=font_size+2, fontweight='bold')
            ax7.set_ylabel('Normalized Value (0=Min, 1=Max)', fontsize=font_size+2, fontweight='bold')
            ax7.legend(fontsize=font_size+1, frameon=True, fancybox=True, shadow=True)
            ax7.grid(True, alpha=0.3)
            plt.tight_layout()
            filename7 = f"{output_dir}/parameter_ranges_parallel_coords.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename7, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename7, dpi=300, bbox_inches='tight')
            plt.close(fig7)
            
            # 8. Search space coverage by generation
            fig8, ax8 = plt.subplots(figsize=(12, 8))
            gen_stats = self.data.groupby('generation').agg({
                'n_qubits': ['min', 'max', 'nunique'],
                'depth': ['min', 'max', 'nunique']
            })
            generations = gen_stats.index
            ax8_twin = ax8.twinx()
            line1 = ax8.plot(generations, gen_stats[('n_qubits', 'nunique')], 'o-', 
                           label='Unique Qubits', linewidth=3, markersize=10, color='blue')
            line2 = ax8.plot(generations, gen_stats[('depth', 'nunique')], 's-', 
                           label='Unique Depths', linewidth=3, markersize=10, color='green')
            line3 = ax8_twin.plot(generations, gen_stats[('n_qubits', 'max')], '^-', 
                                 label='Max Qubits', linewidth=3, markersize=10, color='red', linestyle='--')
            line4 = ax8_twin.plot(generations, gen_stats[('depth', 'max')], 'v-', 
                                 label='Max Depth', linewidth=3, markersize=10, color='orange', linestyle='--')
            ax8.set_xlabel('Generation', fontsize=font_size+2, fontweight='bold')
            ax8.set_ylabel('Unique Values', fontsize=font_size+2, fontweight='bold', color='black')
            ax8_twin.set_ylabel('Max Value', fontsize=font_size+2, fontweight='bold', color='red')
            ax8.tick_params(axis='y', labelcolor='black', labelsize=font_size)
            ax8_twin.tick_params(axis='y', labelcolor='red', labelsize=font_size)
            ax8.tick_params(axis='x', labelsize=font_size)
            lines = line1 + line2 + line3 + line4
            labels = [l.get_label() for l in lines]
            ax8.legend(lines, labels, loc='upper left', fontsize=font_size, frameon=True, fancybox=True, shadow=True)
            ax8.grid(True, alpha=0.3)
            plt.tight_layout()
            filename8 = f"{output_dir}/parameter_ranges_evolution.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename8, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename8, dpi=300, bbox_inches='tight')
            plt.close(fig8)
            
            # 9. Parameter space coverage statistics
            fig9, ax9 = plt.subplots(figsize=(12, 8))
            ax9.axis('off')
            coverage_text = f"""
            SEARCH SPACE COVERAGE
            
            Possible Configurations:
            • Qubits: {len(param_stats['n_qubits']['unique'])} values
            • Depths: {len(param_stats['depth']['unique'])} values
            • Embeddings: {len(param_stats['embed']['unique'])} types
            • Total possible: {total_possible_configs}
            
            Explored Configurations:
            • Unique combinations: {explored_configs}
            • Coverage: {coverage_pct:.1f}%
            • Total evaluations: {len(self.data)}
            
            Parameter Space Dimensions:
            • Qubits range: {param_stats['n_qubits']['min']:.0f} - {param_stats['n_qubits']['max']:.0f}
            • Depth range: {param_stats['depth']['min']:.0f} - {param_stats['depth']['max']:.0f}
            • LR range: {param_stats['learning_rate']['min']:.2e} - {param_stats['learning_rate']['max']:.2e}
            """
            ax9.text(0.1, 0.5, coverage_text, transform=ax9.transAxes, 
                    fontsize=font_size+1, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            plt.tight_layout()
            filename9 = f"{output_dir}/parameter_ranges_coverage_stats.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename9, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename9, dpi=300, bbox_inches='tight')
            plt.close(fig9)
            
            # 10. Multi-dimensional parameter space (3D scatter)
            fig10 = plt.figure(figsize=(12, 10))
            ax10 = fig10.add_subplot(111, projection='3d')
            scatter = ax10.scatter(self.data['n_qubits'], self.data['depth'],
                                  np.log10(self.data['learning_rate']),
                                  c=self.data['val_acc'], cmap='RdYlGn', 
                                  alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            ax10.set_xlabel('Number of Qubits', fontsize=font_size+1, fontweight='bold', labelpad=15)
            ax10.set_ylabel('Circuit Depth', fontsize=font_size+1, fontweight='bold', labelpad=15)
            ax10.set_zlabel('Log10(Learning Rate)', fontsize=font_size+1, fontweight='bold', labelpad=15)
            cbar10 = plt.colorbar(scatter, ax=ax10, shrink=0.6, pad=0.1)
            cbar10.set_label('Accuracy (%)', fontsize=font_size+1)
            cbar10.ax.tick_params(labelsize=font_size)
            plt.tight_layout()
            filename10 = f"{output_dir}/parameter_ranges_3d_space.{file_ext}"
            if file_ext == "svg":
                plt.savefig(filename10, format='svg', bbox_inches='tight')
            else:
                plt.savefig(filename10, dpi=300, bbox_inches='tight')
            plt.close(fig10)
            
            print(f"[OK] Parameter ranges and explored space plots saved:")
            print(f"   - {filename1}")
            print(f"   - {filename2}")
            print(f"   - {filename3}")
            print(f"   - {filename4}")
            print(f"   - {filename5}")
            print(f"   - {filename6}")
            print(f"   - {filename7}")
            print(f"   - {filename8}")
            print(f"   - {filename9}")
            print(f"   - {filename10}")
            print(f"   Coverage: {coverage_pct:.1f}% ({explored_configs}/{total_possible_configs} configurations)")

    def plot_circuit_cuts(self, n_qubits=None, depth=None, ent_ranges=None, cnot_modes=None, 
                         cut_target_qubits=None, eval_id=None, output_dir="outputs/figures", font_size=12, file_ext="png"):
        """Visualize where wire cuts are placed in a quantum circuit.
        
        Parameters:
        -----------
        n_qubits : int, optional
            Number of qubits. If None, uses the best performing configuration from data.
        depth : int, optional
            Circuit depth. If None, uses the best performing configuration from data.
        ent_ranges : list, optional
            Entanglement ranges per layer. If None, uses the best performing configuration from data.
        cnot_modes : list, optional
            CNOT modes per layer. If None, uses the best performing configuration from data.
        cut_target_qubits : int, optional
            Maximum qubits per subcircuit. If None, uses CUT_TARGET_QUBITS from config (default from .env)
        eval_id : str, optional
            Evaluation ID to use for configuration. If provided, overrides other parameters.
        output_dir : str, default="plots"
            Directory to save the plot
        font_size : int, default=12
            Base font size for text elements
        file_ext : str, default="png"
            File extension for saved plots ("png" or "svg")
        """
        # Use CUT_TARGET_QUBITS from config if not specified
        if cut_target_qubits is None:
            cut_target_qubits = CUT_TARGET_QUBITS if CUT_TARGET_QUBITS > 0 else 5
        
        # Import cutting functions
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # Import from new modular structure
        from qnas.quantum.metrics import _qlayer_to_wirecut_string
        from qnas.quantum.circuits import entangling_pairs, _filter_pairs_by_mode
        from qnas.utils.cutter import cut_placement, _suppress_stderr
        
        # Get configuration from data if not provided
        if eval_id and self.data is not None:
            row = self.data[self.data['eval_id'] == eval_id]
            if len(row) > 0:
                n_qubits = int(row.iloc[0]['n_qubits'])
                depth = int(row.iloc[0]['depth'])
                ent_ranges = [int(x) for x in row.iloc[0]['ent_ranges'].split('-')]
                cnot_modes_str = row.iloc[0]['cnot_modes']
                # Convert CNOT mode strings to integers
                mode_map = {'all': 0, 'odd': 1, 'even': 2, 'none': 3}
                cnot_modes = [mode_map.get(m, 0) for m in cnot_modes_str.split('-')]
        elif n_qubits is None and self.data is not None:
            # Use best performing configuration
            best_idx = self.data['val_acc'].idxmax()
            best = self.data.loc[best_idx]
            n_qubits = int(best['n_qubits'])
            depth = int(best['depth'])
            ent_ranges = [int(x) for x in best['ent_ranges'].split('-')]
            cnot_modes_str = best['cnot_modes']
            mode_map = {'all': 0, 'odd': 1, 'even': 2, 'none': 3}
            cnot_modes = [mode_map.get(m, 0) for m in cnot_modes_str.split('-')]
        else:
            # Use defaults if nothing provided
            if n_qubits is None:
                n_qubits = 7
            if depth is None:
                depth = 2
            if ent_ranges is None:
                ent_ranges = [4, 5]
            if cnot_modes is None:
                cnot_modes = [1, 1]  # odd-odd
        
        # Generate circuit string
        qtxt = _qlayer_to_wirecut_string(n_qubits, depth, ent_ranges, cnot_modes)
        
        # Apply cutting
        with _suppress_stderr():
            try:
                cut_circuit, subwires_list = cut_placement(qtxt, cut_target_qubits)
            except Exception as e:
                print(f"⚠ Error in cutting: {e}")
                cut_circuit = qtxt
                subwires_list = [[i for i in range(n_qubits)]]
        
        # Parse the circuit to visualize
        circuit_lines = cut_circuit.split('\n')
        circuit_lines = [l for l in circuit_lines if l.strip()]  # Remove empty lines
        num_cuts = cut_circuit.count('CUT HERE')
        
        # Create visualization with graphical circuit and legend
        fig = plt.figure(figsize=(max(20, num_cuts * 4 + n_qubits * 2), max(8, n_qubits * 1.5)))
        
        # Create grid: 80% for circuit, 20% for legend
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[0, 1])
        
        # Draw circuit diagram with cuts and legend
        self._draw_circuit_with_cuts(ax_main, ax_legend, circuit_lines, n_qubits, subwires_list, 
                                     eval_id if eval_id else 'custom', depth, cut_target_qubits, font_size)
            
        plt.tight_layout()
        filename = f"{output_dir}/circuit_cuts_visualization.{file_ext}"
        if file_ext == "svg":
            plt.savefig(filename, format='svg', bbox_inches='tight')
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[OK] Circuit cuts visualization saved to: {filename}")
        print(f"   Configuration: {n_qubits}q, depth {depth}, {len(subwires_list)} subcircuits")
        print(f"   Cuts placed: {num_cuts}")

    def _draw_circuit_with_cuts(self, ax_main, ax_legend, circuit_lines, n_qubits, subwires_list, eval_id, depth, cut_target_qubits, font_size=10):
        """Draw a CLEAR visual circuit diagram showing wire cuts and subcircuit assignments.
        
        REDESIGNED for maximum clarity:
        - Each subcircuit shown as a separate horizontal block
        - Clean flow from left to right: Input → Subcircuit boxes → Output
        - Wire cuts shown as simple scissors between subcircuits
        - No overlapping colored regions
        - Clear qubit assignment per subcircuit
        """
        import re
        
        # Color scheme for subcircuits (bright, distinct colors with good contrast)
        subcircuit_colors = [
            ('#2E7D32', '#E8F5E9', 'SC1'),  # Green (border, fill, label)
            ('#1565C0', '#E3F2FD', 'SC2'),  # Blue
            ('#E65100', '#FFF3E0', 'SC3'),  # Orange
            ('#6A1B9A', '#F3E5F5', 'SC4'),  # Purple
            ('#C62828', '#FFEBEE', 'SC5'),  # Red
            ('#00695C', '#E0F2F1', 'SC6'),  # Teal
            ('#F9A825', '#FFFDE7', 'SC7'),  # Yellow
            ('#37474F', '#ECEFF1', 'SC8'),  # Blue-grey
        ]
        
        num_subcircuits = len(subwires_list)
        
        # Layout configuration - simplified block-based approach
        block_width = 4.0
        block_height_per_qubit = 0.8
        block_spacing = 3.0  # Space between subcircuit blocks
        margin_left = 4.0
        margin_top = 3.0
        
        total_width = margin_left + num_subcircuits * (block_width + block_spacing) + 2
        total_height = n_qubits * 1.0 + margin_top + 3
        
        ax_main.set_xlim(-1, total_width + 1)
        ax_main.set_ylim(-1, total_height + 1)
        ax_main.set_aspect('equal')
        ax_main.axis('off')
        
        # Wire y-positions (q0 at top)
        wire_y = {i: total_height - margin_top - i * 1.0 for i in range(n_qubits)}
        
        # STEP 1: Draw input qubit labels on the far left
        ax_main.text(1.0, total_height - 1.5, 'INPUT', ha='center', va='center',
                    fontsize=font_size, fontweight='bold', color='#424242')
        
        for i in range(n_qubits):
            y = wire_y[i]
            ax_main.text(1.0, y, f'q[{i}]', ha='center', va='center',
                        fontsize=font_size, fontweight='bold', fontfamily='monospace',
                        color='#424242')
            # Short wire leading to first subcircuit
            ax_main.plot([1.8, margin_left - 0.3], [y, y], 'k-', linewidth=1.5, alpha=0.6)
        
        # STEP 2: Draw each subcircuit as a distinct block
        for sc_idx, subwires in enumerate(subwires_list):
            border_color, fill_color, sc_label = subcircuit_colors[sc_idx % len(subcircuit_colors)]
            
            block_x = margin_left + sc_idx * (block_width + block_spacing)
            
            # Determine block vertical extent based on qubits
            min_q = min(subwires)
            max_q = max(subwires)
            block_top = wire_y[min_q] + 0.5
            block_bottom = wire_y[max_q] - 0.5
            block_h = block_top - block_bottom
            
            # Draw main subcircuit block
            rect = plt.Rectangle(
                (block_x, block_bottom), block_width, block_h,
                facecolor=fill_color, edgecolor=border_color, linewidth=3,
                alpha=0.95, zorder=5
            )
            ax_main.add_patch(rect)
            
            # Subcircuit header (title bar)
            header_h = 0.6
            header_rect = plt.Rectangle(
                (block_x, block_top - header_h), block_width, header_h,
                facecolor=border_color, edgecolor=border_color, linewidth=1,
                alpha=1.0, zorder=6
            )
            ax_main.add_patch(header_rect)
            
            ax_main.text(block_x + block_width/2, block_top - header_h/2,
                        f'SUBCIRCUIT {sc_idx + 1}',
                        ha='center', va='center', fontsize=font_size + 1,
                        fontweight='bold', color='white', zorder=7)
            
            # List qubits inside the block
            sorted_qubits = sorted(subwires)
            qubit_text = ', '.join([f'q{q}' for q in sorted_qubits])
            ax_main.text(block_x + block_width/2, block_bottom + (block_h - header_h)/2,
                        f'Qubits: {qubit_text}\n({len(subwires)} qubits)',
                        ha='center', va='center', fontsize=font_size - 1,
                        fontfamily='monospace', color=border_color, zorder=7)
            
            # Draw wires entering and exiting the block for qubits in this subcircuit
            for q in subwires:
                y = wire_y[q]
                # Incoming wire
                if sc_idx == 0:
                    ax_main.plot([margin_left - 0.3, block_x], [y, y], 
                                'k-', linewidth=1.5, alpha=0.6, zorder=3)
                else:
                    prev_block_end = margin_left + (sc_idx - 1) * (block_width + block_spacing) + block_width
                    ax_main.plot([prev_block_end, block_x], [y, y],
                                'k--', linewidth=1.5, alpha=0.4, zorder=3)
                
                # Outgoing wire
                if sc_idx < num_subcircuits - 1:
                    next_block_start = margin_left + (sc_idx + 1) * (block_width + block_spacing)
                    # Show wire continues (will be drawn by next block)
                else:
                    # Final output wire
                    ax_main.plot([block_x + block_width, block_x + block_width + 1.5], [y, y],
                                'k-', linewidth=1.5, alpha=0.6, zorder=3)
            
            # Draw CUT marker between subcircuits (scissors icon)
            if sc_idx < num_subcircuits - 1:
                cut_x = block_x + block_width + block_spacing / 2
                cut_y_top = wire_y[0] + 0.8
                cut_y_bottom = wire_y[n_qubits - 1] - 0.8
                
                # Dashed cut line
                ax_main.plot([cut_x, cut_x], [cut_y_bottom, cut_y_top],
                            color='#D32F2F', linewidth=3, linestyle='--', zorder=10)
                
                # Scissors icon at top
                ax_main.text(cut_x, cut_y_top + 0.3, '✂', ha='center', va='bottom',
                           fontsize=font_size + 8, color='#D32F2F', zorder=15)
                
                # CUT label
                ax_main.text(cut_x, cut_y_top + 0.9, f'CUT {sc_idx + 1}',
                           ha='center', va='bottom', fontsize=font_size,
                           fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#D32F2F',
                                   edgecolor='#B71C1C', linewidth=2), zorder=15)
        
        # STEP 3: Output labels on the right
        output_x = margin_left + num_subcircuits * (block_width + block_spacing) - block_spacing + 2.5
        ax_main.text(output_x, total_height - 1.5, 'OUTPUT', ha='center', va='center',
                    fontsize=font_size, fontweight='bold', color='#424242')
        
        for i in range(n_qubits):
            y = wire_y[i]
            ax_main.text(output_x, y, f'q[{i}]', ha='center', va='center',
                        fontsize=font_size, fontweight='bold', fontfamily='monospace',
                        color='#424242')
        
        # Title
        ax_main.text(total_width / 2, total_height + 0.3,
                    f'CIRCUIT CUTTING DIAGRAM: {eval_id}',
                    ha='center', va='bottom', fontsize=font_size + 6, fontweight='bold')
        
        subtitle = f'{n_qubits} qubits | {depth} layers | {num_subcircuits} subcircuits | max {cut_target_qubits} qubits per subcircuit'
        ax_main.text(total_width / 2, total_height - 0.3,
                    subtitle,
                    ha='center', va='bottom', fontsize=font_size, color='#616161')
        
        # LEGEND (right panel) - simplified and clear
        ax_legend.axis('off')
        ax_legend.set_xlim(0, 10)
        ax_legend.set_ylim(0, 12)
        
        # Legend title
        ax_legend.text(5, 11.5, 'LEGEND', ha='center', va='top',
                      fontsize=font_size + 2, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#EEEEEE',
                              edgecolor='#757575', linewidth=2))
        
        y_leg = 9.5
        
        # Subcircuit block example
        rect = plt.Rectangle((1.0, y_leg - 0.4), 2.5, 0.8,
                            facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
        ax_legend.add_patch(rect)
        ax_legend.text(2.25, y_leg, 'SC', ha='center', va='center',
                      fontsize=font_size - 1, fontweight='bold', color='#1565C0')
        ax_legend.text(4.5, y_leg, 'Subcircuit\n(Independent\nquantum unit)', ha='left', va='center',
                      fontsize=font_size - 2)
        
        y_leg -= 2.0
        
        # Cut line example
        ax_legend.plot([1.5, 1.5], [y_leg - 0.4, y_leg + 0.4],
                      color='#D32F2F', linewidth=3, linestyle='--')
        ax_legend.text(1.5, y_leg + 0.7, '✂', ha='center', va='center',
                      fontsize=font_size + 4, color='#D32F2F')
        ax_legend.text(4.5, y_leg, 'Wire Cut\n(Partition\nboundary)', ha='left', va='center',
                      fontsize=font_size - 2, color='#D32F2F', fontweight='bold')
        
        y_leg -= 2.5
        
        # Color key for subcircuits
        ax_legend.text(5, y_leg + 0.5, 'SUBCIRCUIT COLORS', ha='center', va='center',
                      fontsize=font_size - 1, fontweight='bold')
        
        for sc_idx, subwires in enumerate(subwires_list[:6]):  # Show max 6
            y_leg -= 0.8
            border_color, fill_color, _ = subcircuit_colors[sc_idx % len(subcircuit_colors)]
            
            # Color swatch
            rect = plt.Rectangle((0.5, y_leg - 0.25), 2.0, 0.5,
                                facecolor=fill_color, edgecolor=border_color, linewidth=2)
            ax_legend.add_patch(rect)
            ax_legend.text(1.5, y_leg, f'SC{sc_idx+1}', ha='center', va='center',
                          fontsize=font_size - 2, fontweight='bold', color=border_color)
            
            # Qubit list
            qubit_str = ', '.join([f'q{q}' for q in sorted(subwires)[:4]])
            if len(subwires) > 4:
                qubit_str += '...'
            ax_legend.text(3.0, y_leg, f'[{qubit_str}]', ha='left', va='center',
                          fontsize=font_size - 2, fontfamily='monospace')

    def plot_circuit_cuts_all_evals(self, cut_target_qubits=None, output_dir="outputs/circuit_cuts", 
                                    font_size=10, file_ext="png"):
        """Generate circuit cuts visualizations for all evaluations in the dataset.
        
        Parameters:
        -----------
        cut_target_qubits : int, optional
            Maximum qubits per subcircuit. If None, uses CUT_TARGET_QUBITS from config (default from .env)
        output_dir : str, default="outputs/circuit_cuts"
            Directory to save all the plots
        font_size : int, default=10
            Base font size for text elements (smaller for many plots)
        file_ext : str, default="png"
            File extension for saved plots ("png" or "svg")
        """
        # Use CUT_TARGET_QUBITS from config if not specified
        if cut_target_qubits is None:
            cut_target_qubits = CUT_TARGET_QUBITS if CUT_TARGET_QUBITS > 0 else 5
        
        if self.data is None or len(self.data) == 0:
            print("[ERROR] No data loaded. Cannot generate circuit cuts visualizations.")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating circuit cuts visualizations for {len(self.data)} evaluations...")
        print(f"Saving to: {output_dir}")
        
        # Import cutting functions once
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # Import from new modular structure
        from qnas.quantum.metrics import _qlayer_to_wirecut_string
        from qnas.utils.cutter import cut_placement, _suppress_stderr
        
        mode_map = {'all': 0, 'odd': 1, 'even': 2, 'none': 3}
        
        success_count = 0
        error_count = 0
        
        for idx, row in self.data.iterrows():
            try:
                eval_id = row['eval_id']
                n_qubits = int(row['n_qubits'])
                depth = int(row['depth'])
                ent_ranges = [int(x) for x in str(row['ent_ranges']).split('-')]
                cnot_modes_str = str(row['cnot_modes'])
                cnot_modes = [mode_map.get(m, 0) for m in cnot_modes_str.split('-')]
                f3 = int(row['f3_n_subcircuits']) if 'f3_n_subcircuits' in row else None
                
                # Generate circuit string
                qtxt = _qlayer_to_wirecut_string(n_qubits, depth, ent_ranges, cnot_modes)
                
                # Apply cutting
                with _suppress_stderr():
                    try:
                        cut_circuit, subwires_list = cut_placement(qtxt, cut_target_qubits)
                        num_subcircuits = len(subwires_list)
                        # Note: F3 mismatch warnings are suppressed - visualization uses current cut_target_qubits,
                        # which may differ from the value used during NSGA-II optimization (stored in CSV)
                        num_cuts = cut_circuit.count('CUT HERE')
                        circuit_lines = cut_circuit.split('\n')
                        circuit_lines = [l for l in circuit_lines if l.strip()]  # Remove empty lines
                    except Exception as e:
                        print(f"⚠ Error cutting circuit for {eval_id}: {e}")
                        cut_circuit = qtxt
                        subwires_list = [[i for i in range(n_qubits)]]
                        num_cuts = 0
                        circuit_lines = cut_circuit.split('\n')
                        circuit_lines = [l for l in circuit_lines if l.strip()]
                
                # Create visualization with graphical circuit and legend
                fig = plt.figure(figsize=(max(20, num_cuts * 4 + n_qubits * 2), max(8, n_qubits * 1.5)))
                
                # Create grid: 80% for circuit, 20% for legend
                gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)
                ax_main = fig.add_subplot(gs[0, 0])
                ax_legend = fig.add_subplot(gs[0, 1])
                
                # Draw circuit diagram with cuts and legend
                self._draw_circuit_with_cuts(ax_main, ax_legend, circuit_lines, n_qubits, subwires_list, 
                                             eval_id, depth, cut_target_qubits, font_size)
                
                plt.tight_layout()
                    
                # Save figure
                safe_eval_id = eval_id.replace('/', '_').replace('\\', '_')
                filename = output_path / f"cuts_{safe_eval_id}.{file_ext}"
                if file_ext == "svg":
                    plt.savefig(filename, format='svg', bbox_inches='tight')
                else:
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                    
                success_count += 1
                if success_count % 10 == 0:
                    print(f"  Generated {success_count}/{len(self.data)} visualizations...")
                        
            except Exception as e:
                error_count += 1
                print(f"  [ERROR] Error processing {row.get('eval_id', 'unknown')}: {e}")
                continue
        
        print(f"\n[OK] Circuit cuts visualizations complete!")
        print(f"  Success: {success_count}")
        if error_count > 0:
            print(f"   [ERROR] Errors: {error_count}")
        print(f"   Saved to: {output_dir}")


def main():
    """Main entry point - generates all analysis figures."""
    import sys
    
    log_section("QNAS Analysis Suite")
    
    # Parse command line arguments
    config = PlotConfig()
    log_folder = None
    dataset = None
    run_dir = None
    weights_path = Path("weights/hybrid_qnn_best_angle-x_nq8_d2.pt")
    
    for arg in sys.argv[1:]:
        if arg == "--svg":
            config.file_ext = "svg"
        elif arg.startswith("--folder="):
            log_folder = arg.split("=")[1]
        elif arg.startswith("--dataset="):
            dataset = arg.split("=")[1]
        elif arg.startswith("--run-dir="):
            run_dir = arg.split("=")[1]
        elif arg.startswith("--weights="):
            weights_path = Path(arg.split("=")[1])
        elif arg.startswith("--font-size="):
            try:
                config.font_size = int(arg.split("=")[1])
            except ValueError:
                print("  [WARN] Invalid font size, using default")
        elif arg.startswith("--output="):
            config.output_dir = Path(arg.split("=")[1])
        elif arg == "list":
            folders = NSGAPlotter.list_log_folders()
            print("\nAvailable log folders:")
            for f in folders:
                print(f"  - {f}")
            return
        elif arg in ("help", "--help", "-h"):
            print(__doc__)
            return
    
    # Update global config
    global PLOT_CONFIG
    PLOT_CONFIG = config
    
    print(f"  Output format: {config.file_ext.upper()}")
    print(f"  Font size: {config.font_size}")
    print(f"  Output dir: {config.output_dir}")
    
    # Auto-detect log folder if not specified
    if not log_folder:
        folders = NSGAPlotter.list_log_folders()
        if len(folders) > 1:
            print("\n  Multiple log folders found:")
            for i, folder in enumerate(folders, 1):
                print(f"    {i}. {folder}")
            print("  Using first folder. Specify with --folder=PATH")
            log_folder = folders[0] if folders else None
        elif folders:
            log_folder = folders[0]
    
    # Initialize plotter
    if log_folder:
        plotter = NSGAPlotter(log_folder=log_folder, dataset=dataset, run_dir=run_dir)
    else:
        plotter = NSGAPlotter()
    
    if plotter.data is None:
        print("\n  [ERROR] Could not load data")
        if log_folder:
            print(f"  Check path: logs/{log_folder}/")
        return
    
    # Determine log directory
    if log_folder and dataset and run_dir:
        log_dir = Path(f"logs/{log_folder}/{dataset}/{run_dir}")
    elif hasattr(plotter, 'run_dir') and plotter.run_dir:
        log_dir = plotter.run_dir
        print(f"  Run directory: {log_dir}")
    elif log_folder == "nsga-ii":
        base_path = Path("logs/nsga-ii")
        if dataset:
            dataset_path = base_path / dataset
        else:
            datasets = [d.name for d in base_path.iterdir() if d.is_dir()]
            if datasets:
                dataset = datasets[0]
                dataset_path = base_path / dataset
            else:
                dataset_path = base_path
        runs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)
        log_dir = runs[0] if runs else dataset_path
    else:
        log_dir = plotter.run_dir if hasattr(plotter, 'run_dir') and plotter.run_dir else Path("logs")
    
    # Set output directory to match run folder name
    # Extract run folder name from various possible paths
    # Examples: 
    #   - "logs/run_20251227-003334" -> "run_20251227-003334"
    #   - "logs/nsga-ii/MNIST/run_20251227-003334" -> "run_20251227-003334"
    #   - "run_20251227-003334" -> "run_20251227-003334"
    run_folder_name = "default"  # fallback if no run folder detected
    if log_dir.name.startswith('run_'):
        run_folder_name = log_dir.name
    elif log_dir.parent.name.startswith('run_'):
        run_folder_name = log_dir.parent.name
    elif log_folder and log_folder.startswith('run_'):
        run_folder_name = log_folder
    elif log_folder and '/' in log_folder:
        # Handle "logs/run_20251227-003334" format
        parts = log_folder.split('/')
        for part in reversed(parts):
            if part.startswith('run_'):
                run_folder_name = part
                break
    
    # Only override if user didn't specify custom output
    if not any(arg.startswith("--output=") for arg in sys.argv[1:]):
        config.output_dir = Path("outputs") / run_folder_name / "figures"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output dir: {config.output_dir} (matched to run folder)")
    
    # Update global config
    PLOT_CONFIG = config
    
    log_section("Generating Figures")
    
    # Step 1: Core evolution figures
    log_step(1, "Evolution and Pareto figures")
    generate_core_figures(plotter, log_dir, config)
    
    # Step 2: Hyperparameter analysis
    log_step(2, "Hyperparameter analysis")
    plotter.plot_hyperparameter_analysis(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    plotter.plot_performance_distributions(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    plotter.plot_best_performers(str(config.output_dir), top_n=10, font_size=config.font_size, file_ext=config.file_ext)
    
    # Step 3: F3 analysis
    log_step(3, "F3 (subcircuits) analysis")
    original_create = plotter.create_output_dir
    plotter.create_output_dir = lambda x: str(config.output_dir)
    try:
        plotter.generate_f3_focused_plots(file_ext=config.file_ext)
        plotter.analyze_f3_insights()
    finally:
        plotter.create_output_dir = original_create
    
    # Step 4: 3D visualizations
    log_step(4, "3D visualizations")
    try:
        plotter.plot_3d_accuracy_f2_f3(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] 3D plots: {e}")
    
    # Step 5: Correlation analysis
    log_step(5, "Correlation analysis")
    try:
        plotter.plot_f1_f3_correlation(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] F1-F3 correlation: {e}")
    
    # Step 6: Pareto with F3
    log_step(6, "Pareto fronts with F3")
    try:
        plotter.plot_pareto_front_f3(str(config.output_dir), file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] Pareto F3: {e}")
    
    # Step 7: Dashboard
    log_step(7, "Comprehensive dashboard")
    try:
        plotter.plot_comprehensive_summary(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] Dashboard: {e}")
    
    # Step 8: Merged figures
    log_step(8, "Merged multi-panel figures")
    merge_figures(config.output_dir, config.output_dir, config.output_dir, config.file_ext)
    
    # Step 9: Search space
    log_step(9, "Search space visualizations")
    try:
        plotter.plot_search_space_exploration(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
        plotter.plot_parameter_space_coverage(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
        plotter.plot_hyperparameter_combinations(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
        plotter.plot_parameter_ranges_explored(str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] Search space: {e}")
    
    # Step 10: Training curves
    log_step(10, "Training curves")
    try:
        plotter.plot_pareto_training_curves(output_dir=str(config.output_dir), font_size=config.font_size, file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] Training curves: {e}")
    
    # Step 11: Circuit cuts
    log_step(11, "Circuit cut visualizations")
    try:
        if plotter.data is not None and len(plotter.data) > 0:
            # circuit_cuts goes at run folder level, not under figures/
            cuts_dir = config.output_dir.parent / "circuit_cuts"
            plotter.plot_circuit_cuts_all_evals(output_dir=str(cuts_dir), font_size=10, file_ext=config.file_ext)
    except Exception as e:
        print(f"  [SKIP] Circuit cuts: {e}")
    
    log_section("Complete")
    print(f"  All figures saved to: {config.output_dir}")
    print(f"  Circuit cuts saved to: {config.output_dir.parent / 'circuit_cuts'}")
    print()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_generation_data(g):
    """Prepare generation summary data."""
    g = g.copy()
    if "best_1_minus_acc" in g.columns:
        g["best_accuracy"] = (1.0 - g["best_1_minus_acc"]) * 100.0
    if "median_1_minus_acc" in g.columns:
        g["median_accuracy"] = (1.0 - g["median_1_minus_acc"]) * 100.0
    if "generation" not in g.columns:
        g["generation"] = np.arange(len(g))
    return g


# ============================================================================
# PAPER FIGURE GENERATION FUNCTIONS
# ============================================================================

def categorize_cnot_mode(cnot_str: str) -> str:
    """Categorize CNOT mode string into Sparse/Mixed/Dense."""
    if pd.isna(cnot_str) or cnot_str == "":
        return "Sparse (none)"
    
    cnot_str = str(cnot_str).lower()
    parts = [p.strip() for p in cnot_str.split("-") if p.strip()]
    
    # Check if all are "none"
    if all(p == "none" for p in parts):
        return "Sparse (none)"
    
    # Check if any are "all"
    if any(p == "all" for p in parts):
        return "Dense (all)"
    
    # Check if mix of even/odd/none
    has_even = any(p == "even" for p in parts)
    has_odd = any(p == "odd" for p in parts)
    has_none = any(p == "none" for p in parts)
    
    if (has_even or has_odd) and not has_none:
        return "Mixed (even/odd)"
    elif has_none and (has_even or has_odd):
        return "Mixed (even/odd)"  # Mixed with some none
    else:
        return "Sparse (none)"  # Default to sparse


def parse_progress_log(log_path: Path):
    """Parse progress.log to extract final training curve."""
    if not log_path.exists():
        return None
    
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if '[GPU0|final-best]' in line and 'Ep' in line:
                match = re.search(r'Ep (\d+)/\d+', line)
                if match:
                    epoch = int(match.group(1))
                    train_match = re.search(r'train_loss=([\d.]+)', line)
                    train_loss = float(train_match.group(1)) if train_match else None
                    train_acc_match = re.search(r'train.*acc=([\d.]+)%', line)
                    train_acc = float(train_acc_match.group(1)) if train_acc_match else None
                    val_match = re.search(r'val_loss=([\d.]+)', line)
                    val_loss = float(val_match.group(1)) if val_match else None
                    val_acc_match = re.search(r'val.*acc=([\d.]+)%', line)
                    val_acc = float(val_acc_match.group(1)) if val_acc_match else None
                    
                    if all(x is not None for x in [train_loss, train_acc, val_loss, val_acc]):
                        epochs.append(epoch)
                        train_losses.append(train_loss)
                        train_accs.append(train_acc)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
    
    if not epochs:
        return None
    
    return pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })


def find_pareto_front_2d(df, x_col, y_col, x_minimize=True, y_minimize=False):
    """
    Find Pareto-optimal points in 2D space.
    
    Args:
        df: DataFrame with data points
        x_col: Column name for x-axis (e.g., 'f2_circuit_cost')
        y_col: Column name for y-axis (e.g., 'val_acc')
        x_minimize: True if we want to minimize x (default: True for cost)
        y_minimize: False if we want to maximize y (default: False for accuracy)
    
    Returns:
        DataFrame with Pareto-optimal points, sorted for plotting
    """
    # Remove NaN values
    valid_mask = ~(df[x_col].isna() | df[y_col].isna())
    data = df[valid_mask].copy()
    
    if len(data) == 0:
        return pd.DataFrame()
    
    # Convert to numpy for faster computation
    x_vals = data[x_col].values
    y_vals = data[y_col].values
    
    pareto_mask = []
    for i in range(len(data)):
        is_dominated = False
        for j in range(len(data)):
            if i == j:
                continue
            
            # Check if point j dominates point i
            # For minimization: lower is better
            # For maximization: higher is better
            
            # Normalize: convert to minimization problem
            if x_minimize:
                x_i, x_j = x_vals[i], x_vals[j]
            else:
                x_i, x_j = -x_vals[i], -x_vals[j]  # Negate to convert to minimization
            
            if y_minimize:
                y_i, y_j = y_vals[i], y_vals[j]
            else:
                y_i, y_j = -y_vals[i], -y_vals[j]  # Negate to convert to minimization
            
            # Now both are minimization problems
            # Point j dominates point i if:
            # - x_j <= x_i AND y_j <= y_i (j is not worse in any objective)
            # - AND at least one inequality is strict (j is better in at least one)
            if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                is_dominated = True
                break
        
        pareto_mask.append(not is_dominated)
    
    pareto_df = data[pareto_mask].copy()
    
    # Sort by x for smooth line plotting
    if len(pareto_df) > 0:
        pareto_df = pareto_df.sort_values(x_col)
    
    return pareto_df


def generate_core_figures(plotter, log_dir: Path, config: PlotConfig):
    """Generate core evolution and Pareto figures with consistent naming."""
    
    colors = config.colors
    
    # Load data
    gen_summary_path = log_dir / 'nsga_gen_summary.csv'
    if not gen_summary_path.exists():
        print(f"  [WARN] Gen summary not found: {gen_summary_path}")
        return
    
    gen_summary = pd.read_csv(gen_summary_path)
    evals = plotter.data.copy()
    
    def save_fig(fig, name: str):
        """Save figure using global config."""
        save_figure(fig, name, config)
    
    # Evolution of Accuracy
    fig, ax = plt.subplots(figsize=config.figsize_small)
    generations = gen_summary['generation']
    best_acc = (1 - gen_summary['best_1_minus_acc']) * 100
    median_acc = (1 - gen_summary['median_1_minus_acc']) * 100
    ax.plot(generations, best_acc, 'o-', label='Best-of-generation', linewidth=2, markersize=8, color=colors[0])
    ax.plot(generations, median_acc, 's-', label='Median', linewidth=2, markersize=8, color=colors[1])
    ax.set_xlabel('Generation', fontsize=config.font_size)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=config.font_size)
    ax.legend(fontsize=config.font_size-2)
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'evolution_accuracy')
    
    # Evolution of F2
    fig, ax = plt.subplots(figsize=config.figsize_small)
    ax.plot(generations, gen_summary['best_cost'], 'o-', linewidth=2, markersize=8, color=colors[2])
    ax.set_xlabel('Generation', fontsize=config.font_size)
    ax.set_ylabel('Best F2 (s/sample)', fontsize=config.font_size)
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'evolution_f2')
    
    # Pareto Acc vs F2 (plot area 8x3, extra height for legend below)
    fig, ax = plt.subplots(figsize=(8, 4.5))  # Extra 1.5 inches for 2-line legend below
    best_idx = evals['val_acc'].idxmax()
    best_acc = evals.loc[best_idx, 'val_acc']
    best_f2 = evals.loc[best_idx, 'f2_circuit_cost']
    
    # Plot all points
    scatter = ax.scatter(evals['f2_circuit_cost'], evals['val_acc'], c=evals['gen_est'], cmap='viridis', alpha=0.6, s=50, label='All solutions', zorder=1)
    
    # Find Pareto-optimal points (minimize f2, maximize accuracy)
    pareto_df = find_pareto_front_2d(evals, 'f2_circuit_cost', 'val_acc', x_minimize=True, y_minimize=False)
    
    print(f"Found {len(pareto_df)} Pareto-optimal points out of {len(evals)} total points")
    
    if len(pareto_df) > 0:
        # Sort by f2 (cost) for proper line connection
        pareto_df_sorted = pareto_df.sort_values('f2_circuit_cost')
        
        # Plot Pareto front with stepped lines (horizontal then vertical segments)
        ax.step(pareto_df_sorted['f2_circuit_cost'], pareto_df_sorted['val_acc'], 
               where='post', color='red', linewidth=2, alpha=0.8, label='Pareto front', zorder=4)
        
        # Highlight Pareto points
        ax.scatter(pareto_df_sorted['f2_circuit_cost'], pareto_df_sorted['val_acc'], 
                  color='red', s=80, edgecolors='darkred', linewidths=1.5, 
                  marker='o', zorder=5, label='Pareto-optimal', alpha=0.9)
        
        print(f"Pareto front range: f2=[{pareto_df_sorted['f2_circuit_cost'].min():.6f}, {pareto_df_sorted['f2_circuit_cost'].max():.6f}], "
              f"acc=[{pareto_df_sorted['val_acc'].min():.2f}%, {pareto_df_sorted['val_acc'].max():.2f}%]")
    else:
        print("[WARN] Warning: No Pareto-optimal points found!")
    
    # Mark best accuracy point (smaller star)
    ax.scatter([best_f2], [best_acc], marker='*', s=280, color='gold', 
              edgecolors='black', linewidths=2, zorder=6, label=f'Best accuracy: {best_acc:.2f}%')
    
    # Set axis limits with padding so best point and all data are visible
    x_min, x_max = evals['f2_circuit_cost'].min(), evals['f2_circuit_cost'].max()
    y_min, y_max = evals['val_acc'].min(), evals['val_acc'].max()
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 0.1)
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.08 * y_range)
    ax.set_xlabel('F2 Circuit Cost (s/sample)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Val Acc (%)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    # Legend below plot in 2 lines (ncol=2), below x-axis with gap
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=14, frameon=True)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Generation')
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # Reserve bottom 18% for 2-line legend (plot stays ~4.1 inches)
    save_fig(fig, 'pareto_accuracy_vs_f2')
    
    # F3 vs Qubits
    fig, ax = plt.subplots(figsize=(8, 3))  # Half height: 6 -> 3
    valid_mask = ~(evals['n_qubits'].isna() | evals['f3_n_subcircuits'].isna() | evals['seconds'].isna())
    valid_data = evals[valid_mask]
    sc = ax.scatter(valid_data['n_qubits'], valid_data['f3_n_subcircuits'], c=valid_data['seconds'], cmap='viridis', alpha=0.7, s=50)
    # Use CUT_TARGET_QUBITS from config (read from .env), fallback to 5 if not set
    cut_target = CUT_TARGET_QUBITS if CUT_TARGET_QUBITS > 0 else 5
    ax.axvline(cut_target, linestyle='--', color='red', linewidth=2, alpha=0.7, label='')
    ax.set_xlabel('Number of Qubits (n)', fontsize=18, fontweight='bold')
    ax.set_ylabel('F3 (Subcircuits)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, label='Training Time (s)')
    cbar.set_label('Training Time (s)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    save_fig(fig, 'f3_vs_qubits')
    
    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=config.figsize_small)
    cols = ['val_acc', 'f2_circuit_cost', 'f3_n_subcircuits', 'n_qubits', 'depth', 'seconds']
    corr_data = evals[cols].corr()
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, mask=mask, cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_fig(fig, 'correlation_heatmap')
    
    # CNOT Mode Accuracy
    fig, ax = plt.subplots(figsize=config.figsize_small)
    evals['cnot_category'] = evals['cnot_modes'].apply(categorize_cnot_mode)
    category_order = ["Sparse (none)", "Mixed (even/odd)", "Dense (all)"]
    evals_categorized = evals[evals['cnot_category'].isin(category_order)].copy()
    box_data = [evals_categorized[evals_categorized['cnot_category'] == cat]['val_acc'].values for cat in category_order]
    bp = ax.boxplot(box_data, tick_labels=category_order, patch_artist=True)
    colors_box = [colors[0], colors[1], colors[2]]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_xlabel('CNOT Mode Category')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(90, linestyle='--', color='red', linewidth=1.5, alpha=0.7, label='90% threshold')
    ax.legend()
    save_fig(fig, 'cnot_mode_accuracy')
    
    # Figure 7: Final Training Curves (Combined plot with 3 subplots for each Pareto Optimal Point)
    # Print-friendly dark colors and large fonts for visibility
    curve_colors = ['#0d47a1', '#bf360c', '#1b5e20', '#4a148c']  # blue, orange, green, purple
    curve_fs_label = 32
    curve_fs_tick = 24
    curve_fs_title = 34
    curve_fs_legend = 28
    train_log_path = log_dir / 'train_epoch_log.csv'
    if train_log_path.exists():
        try:
            train_data = pd.read_csv(train_log_path)
            # Auto-detect pareto optimal eval_ids
            pareto_ids = train_data[train_data['eval_id'].str.startswith('final-pareto-')]['eval_id'].unique().tolist()
            
            if len(pareto_ids) > 0:
                # Color palette for different pareto points
                pareto_colors = {
                    'final-pareto-best_accuracy': curve_colors[0],
                    'final-pareto-lowest_cost': curve_colors[1],
                    'final-pareto-balanced_2': curve_colors[2],
                }
                default_colors = curve_colors
                
                # First pass: compute shared axis limits across all Pareto points
                all_epochs, all_loss, all_acc = [], [], []
                for eval_id in pareto_ids:
                    eval_data = train_data[train_data['eval_id'] == eval_id].copy()
                    epoch_data = eval_data[eval_data['phase'] == 'epoch_end'].copy()
                    if len(epoch_data) == 0:
                        continue
                    all_epochs.extend(epoch_data['epoch'].dropna().tolist())
                    if 'train_loss' in epoch_data.columns:
                        all_loss.extend(epoch_data['train_loss'].dropna().tolist())
                    if 'val_loss' in epoch_data.columns:
                        all_loss.extend(epoch_data['val_loss'].dropna().tolist())
                    if 'train_acc' in epoch_data.columns:
                        all_acc.extend(epoch_data['train_acc'].dropna().tolist())
                    if 'val_acc' in epoch_data.columns:
                        all_acc.extend(epoch_data['val_acc'].dropna().tolist())
                
                # Compute shared limits with small padding
                epoch_min, epoch_max = min(all_epochs), max(all_epochs)
                loss_min, loss_max = min(all_loss), max(all_loss)
                acc_min, acc_max = min(all_acc), max(all_acc)
                epoch_pad = 0.02 * (epoch_max - epoch_min) if epoch_max > epoch_min else 1
                loss_pad = 0.05 * (loss_max - loss_min) if loss_max > loss_min else 0.1
                acc_pad = 0.05 * (acc_max - acc_min) if acc_max > acc_min else 1
                
                # 3 subplots in a row with shared axis limits
                n_pareto = len(pareto_ids)
                fig, axes = plt.subplots(1, n_pareto, figsize=(config.figsize_wide[0], 5), sharey=False)
                if n_pareto == 1:
                    axes = [axes]
                
                all_lines = []
                all_labels = []
                twin_axes = []  # Store twin axes for later limit setting
                
                for i, eval_id in enumerate(pareto_ids):
                    eval_data = train_data[train_data['eval_id'] == eval_id].copy()
                    epoch_data = eval_data[eval_data['phase'] == 'epoch_end'].copy()
                    epoch_data = epoch_data.sort_values('epoch')
                    if len(epoch_data) == 0:
                        continue
                    
                    label = eval_id.replace('final-pareto-', '').replace('_', ' ').title()
                    
                    ax1 = axes[i]
                    ax1.grid(True, alpha=0.3)
                    ax2 = ax1.twinx()
                    twin_axes.append(ax2)
                    
                    # Plot loss on left axis
                    if 'train_loss' in epoch_data.columns and not epoch_data['train_loss'].isna().all():
                        line1 = ax1.plot(epoch_data['epoch'], epoch_data['train_loss'], 'o-', 
                                        color=curve_colors[0], linewidth=2.5, markersize=6, alpha=0.95, linestyle='--')
                        if i == 0:
                            all_lines.extend(line1)
                            all_labels.append('Train Loss')
                    if 'val_loss' in epoch_data.columns and not epoch_data['val_loss'].isna().all():
                        line2 = ax1.plot(epoch_data['epoch'], epoch_data['val_loss'], 's-', 
                                        color=curve_colors[1], linewidth=2.5, markersize=6, alpha=0.95)
                        if i == 0:
                            all_lines.extend(line2)
                            all_labels.append('Val Loss')
                    # Plot accuracy on right axis
                    if 'train_acc' in epoch_data.columns and not epoch_data['train_acc'].isna().all():
                        line3 = ax2.plot(epoch_data['epoch'], epoch_data['train_acc'], 'o--', 
                                        color=curve_colors[2], linewidth=2.5, markersize=6, alpha=0.95)
                        if i == 0:
                            all_lines.extend(line3)
                            all_labels.append('Train Acc')
                    if 'val_acc' in epoch_data.columns and not epoch_data['val_acc'].isna().all():
                        line4 = ax2.plot(epoch_data['epoch'], epoch_data['val_acc'], 's--', 
                                        color=curve_colors[3], linewidth=2.5, markersize=6, alpha=0.95)
                        if i == 0:
                            all_lines.extend(line4)
                            all_labels.append('Val Acc')
                    
                    # Set shared axis limits
                    ax1.set_xlim(epoch_min - epoch_pad, epoch_max + epoch_pad)
                    ax1.set_ylim(loss_min - loss_pad, loss_max + loss_pad)
                    ax2.set_ylim(acc_min - acc_pad, acc_max + acc_pad)
                    
                    ax1.set_title(f'{label}', fontsize=curve_fs_title, fontweight='bold', pad=10)
                    
                    # Only leftmost subplot shows Loss tick labels and y-label
                    if i == 0:
                        ax1.set_ylabel('Loss', color=curve_colors[0], fontsize=curve_fs_label, fontweight='bold')
                        ax1.tick_params(axis='y', labelcolor=curve_colors[0], labelsize=curve_fs_tick)
                    else:
                        ax1.set_ylabel('')
                        ax1.tick_params(axis='y', labelleft=False)  # Hide tick labels
                    
                    # Only rightmost subplot shows Accuracy tick labels and y-label
                    if i == n_pareto - 1:
                        ax2.set_ylabel('Accuracy (%)', color=curve_colors[2], fontsize=curve_fs_label, fontweight='bold')
                        ax2.tick_params(axis='y', labelcolor=curve_colors[2], labelsize=curve_fs_tick)
                    else:
                        ax2.set_ylabel('')
                        ax2.tick_params(axis='y', labelright=False)  # Hide tick labels
                    
                    ax1.tick_params(axis='x', labelsize=curve_fs_tick)
                
                # Single shared x-label at the bottom center
                fig.text(0.5, 0.02, 'Epoch', ha='center', fontsize=curve_fs_label, fontweight='bold')
                
                # Add single shared legend below plots
                if all_lines:
                    fig.legend(all_lines, all_labels, loc='upper center', ncol=4, fontsize=curve_fs_legend, 
                              frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.12))
                
                plt.tight_layout(rect=[0, 0.16, 1, 1])
                save_fig(fig, 'final_training_curve')
            else:
                # Fallback to progress.log parsing if no pareto points found
                progress_log_path = log_dir / 'progress.log'
                training_data = parse_progress_log(progress_log_path)
                if training_data is not None and len(training_data) > 0:
                    fig, ax1 = plt.subplots(figsize=config.figsize_small)
                    epochs = training_data['epoch']
                    ax1.set_xlabel('Epoch', fontsize=curve_fs_label, fontweight='bold')
                    ax1.set_ylabel('Loss', color=curve_colors[0], fontsize=curve_fs_label, fontweight='bold')
                    line1 = ax1.plot(epochs, training_data['train_loss'], 'o-', label='Train Loss', color=curve_colors[0], linewidth=2.5, markersize=6, alpha=0.95)
                    line2 = ax1.plot(epochs, training_data['val_loss'], 's-', label='Val Loss', color=curve_colors[1], linewidth=2.5, markersize=6, alpha=0.95)
                    ax1.tick_params(axis='y', labelcolor=curve_colors[0], labelsize=curve_fs_tick)
                    ax1.tick_params(axis='x', labelsize=curve_fs_tick)
                    ax1.grid(True, alpha=0.3)
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Accuracy (%)', color=curve_colors[2], fontsize=curve_fs_label, fontweight='bold')
                    line3 = ax2.plot(epochs, training_data['train_acc'], 'o--', label='Train Acc', color=curve_colors[2], linewidth=2.5, markersize=6, alpha=0.95)
                    line4 = ax2.plot(epochs, training_data['val_acc'], 's--', label='Val Acc', color=curve_colors[3], linewidth=2.5, markersize=6, alpha=0.95)
                    ax2.tick_params(axis='y', labelcolor=curve_colors[2], labelsize=curve_fs_tick)
                    lines = line1 + line2 + line3 + line4
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc='center right', fontsize=curve_fs_legend)
                    save_fig(fig, 'final_training_curve')
                else:
                    print("  [WARN] Could not parse final training curve from progress.log")
        except Exception as e:
            print(f"  [WARN] Error loading training curves: {e}")
            # Fallback to progress.log
            progress_log_path = log_dir / 'progress.log'
            training_data = parse_progress_log(progress_log_path)
            if training_data is not None and len(training_data) > 0:
                fig, ax1 = plt.subplots(figsize=config.figsize_small)
                epochs = training_data['epoch']
                ax1.set_xlabel('Epoch', fontsize=curve_fs_label, fontweight='bold')
                ax1.set_ylabel('Loss', color=curve_colors[0], fontsize=curve_fs_label, fontweight='bold')
                line1 = ax1.plot(epochs, training_data['train_loss'], 'o-', label='Train Loss', color=curve_colors[0], linewidth=2.5, markersize=6, alpha=0.95)
                line2 = ax1.plot(epochs, training_data['val_loss'], 's-', label='Val Loss', color=curve_colors[1], linewidth=2.5, markersize=6, alpha=0.95)
                ax1.tick_params(axis='y', labelcolor=curve_colors[0], labelsize=curve_fs_tick)
                ax1.tick_params(axis='x', labelsize=curve_fs_tick)
                ax1.grid(True, alpha=0.3)
                ax2 = ax1.twinx()
                ax2.set_ylabel('Accuracy (%)', color=curve_colors[2], fontsize=curve_fs_label, fontweight='bold')
                line3 = ax2.plot(epochs, training_data['train_acc'], 'o--', label='Train Acc', color=curve_colors[2], linewidth=2.5, markersize=6, alpha=0.95)
                line4 = ax2.plot(epochs, training_data['val_acc'], 's--', label='Val Acc', color=curve_colors[3], linewidth=2.5, markersize=6, alpha=0.95)
                ax2.tick_params(axis='y', labelcolor=curve_colors[2], labelsize=curve_fs_tick)
                lines = line1 + line2 + line3 + line4
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='center right', fontsize=curve_fs_legend)
                save_fig(fig, 'final_training_curve')
    else:
        # Fallback to progress.log if train_epoch_log.csv doesn't exist
        progress_log_path = log_dir / 'progress.log'
        training_data = parse_progress_log(progress_log_path)
        if training_data is not None and len(training_data) > 0:
            # Use same print-friendly styling as main path (large fonts for visibility)
            curve_colors = ['#0d47a1', '#bf360c', '#1b5e20', '#4a148c']
            curve_fs_label, curve_fs_tick, curve_fs_legend = 32, 24, 28
            fig, ax1 = plt.subplots(figsize=config.figsize_small)
            epochs = training_data['epoch']
            ax1.set_xlabel('Epoch', fontsize=curve_fs_label, fontweight='bold')
            ax1.set_ylabel('Loss', color=curve_colors[0], fontsize=curve_fs_label, fontweight='bold')
            line1 = ax1.plot(epochs, training_data['train_loss'], 'o-', label='Train Loss', color=curve_colors[0], linewidth=2.5, markersize=6, alpha=0.95)
            line2 = ax1.plot(epochs, training_data['val_loss'], 's-', label='Val Loss', color=curve_colors[1], linewidth=2.5, markersize=6, alpha=0.95)
            ax1.tick_params(axis='y', labelcolor=curve_colors[0], labelsize=curve_fs_tick)
            ax1.tick_params(axis='x', labelsize=curve_fs_tick)
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy (%)', color=curve_colors[2], fontsize=curve_fs_label, fontweight='bold')
            line3 = ax2.plot(epochs, training_data['train_acc'], 'o--', label='Train Acc', color=curve_colors[2], linewidth=2.5, markersize=6, alpha=0.95)
            line4 = ax2.plot(epochs, training_data['val_acc'], 's--', label='Val Acc', color=curve_colors[3], linewidth=2.5, markersize=6, alpha=0.95)
            ax2.tick_params(axis='y', labelcolor=curve_colors[2], labelsize=curve_fs_tick)
            lines = line1 + line2 + line3 + line4
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right', fontsize=curve_fs_legend)
            save_fig(fig, 'final_training_curve')
        else:
            print("⚠ Could not parse final training curve from progress.log")
    
    # Note: Transpiled circuit generation uses scripts/utils/transpile_circuit.py
    
    print(f"  Core figures saved to: {config.output_dir}")


def merge_figures(figures_dir: Path, plots_dir: Path, output_dir: Path, file_ext: str = "png"):
    """Merge individual figures into multi-panel dashboard layouts."""
    def load_image(img_path: Path):
        if not img_path.exists():
            return None
        try:
            img = Image.open(img_path)
            return np.array(img)
        except Exception as e:
            return None
    
    def save_merged(fig, name: str):
        """Save merged figure."""
        filename = f"dashboard_{name}.{file_ext}"
        if file_ext == "svg":
            fig.savefig(output_dir / filename, format='svg', bbox_inches='tight')
        else:
            fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] {filename}")
    
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standardized figure names (without dataset prefix)
    fig_names = {
        'evolution_accuracy': 'evolution_accuracy',
        'evolution_f2': 'evolution_f2',
        'final_training_curve': 'final_training_curve',  # Combined plot with 3 subplots
        'pareto_accuracy_vs_f2': 'pareto_accuracy_vs_f2',
        'f3_vs_qubits': 'f3_vs_qubits',
        'correlation_heatmap': 'correlation_heatmap',
        'cnot_mode_accuracy': 'cnot_mode_accuracy'
    }
    
    # Comprehensive Dashboard
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.25)
    dashboard_keys = ['evolution_accuracy', 'evolution_f2', 'final_training_curve',
                      'pareto_accuracy_vs_f2', 'cnot_mode_accuracy', 'correlation_heatmap',
                      'f3_vs_qubits']
    for i, key in enumerate(dashboard_keys):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        # Try the mapped name first, then fallback to key name if not found
        img_path = figures_dir / f'{fig_names.get(key, key)}.{file_ext}'
        if not img_path.exists() and key == 'final_training_curve':
            # Try alternative training curve names (fallback for old individual plots)
            for alt_name in ['final_training_curve', 'final_training_curve_best_accuracy', 'final_training_curve_lowest_cost']:
                alt_path = figures_dir / f'{alt_name}.{file_ext}'
                if alt_path.exists():
                    img_path = alt_path
                    break
        img = load_image(img_path)
        if img is not None:
            ax.imshow(img)
            ax.axis('off')
            title = key.replace('_', ' ').title()
    save_merged(fig, 'comprehensive')
    
    print(f"  Merged figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

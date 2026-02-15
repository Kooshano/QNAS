#!/usr/bin/env python3
"""
Analyze correlation between checkpoint accuracies at different epochs and final accuracies.
Determines optimal checkpoint size for predicting final model performance.
Analyzes correlations across all epochs and checkpoint sizes.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
from typing import Optional


def analyze_all_epochs_correlations(checkpoint_file: str, output_dir: Path, final_epoch: Optional[int] = None):
    """
    Analyze correlations between all epochs/checkpoints and the final epoch.
    All correlations are computed against the specified (or max) final epoch.

    Args:
        checkpoint_file: Path to checkpoint_validation.csv file
        output_dir: Directory to save results
        final_epoch: Epoch to use as final accuracy benchmark. If None, uses max(available_epochs).
    """
    print("="*80)
    print("COMPREHENSIVE EPOCH & CHECKPOINT CORRELATION ANALYSIS")
    print("All epochs and checkpoints correlated to FINAL EPOCH")
    print("="*80)
    print(f"Reading data from: {checkpoint_file}\n")

    # Read the checkpoint validation data
    df = pd.read_csv(checkpoint_file)

    # Get all available epochs and checkpoint sizes
    available_epochs = sorted(df['epoch'].unique())
    checkpoint_sizes = sorted(df['checkpoint_train_size'].unique())
    max_checkpoint_size = max(checkpoint_sizes)

    print(f"Available epochs: {available_epochs}")
    print(f"Available checkpoint sizes: {checkpoint_sizes}")
    print(f"Maximum checkpoint size: {max_checkpoint_size}\n")

    # Determine final epoch (user-specified or maximum available)
    if final_epoch is None:
        final_epoch = max(available_epochs)
        print(f"Using maximum available epoch as final: {final_epoch}")
    else:
        if final_epoch not in available_epochs:
            print(f"[ERROR] Final epoch {final_epoch} not in data. Available: {available_epochs}")
            sys.exit(1)
        print(f"Using specified final epoch: {final_epoch}")
    
    # Get final accuracies at final epoch (at max checkpoint size)
    df_final = df[(df['epoch'] == final_epoch) & (df['checkpoint_train_size'] == max_checkpoint_size)]
    if len(df_final) == 0:
        print(f"[ERROR] No data found for final epoch {final_epoch} at max checkpoint size {max_checkpoint_size}")
        return None
    
    final_data_dict = df_final.set_index('eval_id')['val_acc'].to_dict()
    print(f"Final epoch: {final_epoch}")
    print(f"Found final accuracies for {len(final_data_dict)} candidates at epoch {final_epoch}\n")
    
    # Calculate correlations for all checkpoint epochs and sizes to final epoch
    all_results = []
    
    for checkpoint_epoch in available_epochs:
        # Get checkpoint data for this epoch
        df_checkpoint = df[df['epoch'] == checkpoint_epoch].copy()
        
        # Analyze each checkpoint size
        for checkpoint_size in checkpoint_sizes:
            # Skip if comparing same epoch, same size (perfect correlation with itself)
            if checkpoint_epoch == final_epoch and checkpoint_size == max_checkpoint_size:
                continue
            
            checkpoint_data = df_checkpoint[
                df_checkpoint['checkpoint_train_size'] == checkpoint_size
            ][['eval_id', 'val_acc']].set_index('eval_id')['val_acc'].to_dict()
            
            # Find common eval_ids
            common_ids = set(checkpoint_data.keys()) & set(final_data_dict.keys())
            
            if len(common_ids) < 2:
                continue
            
            # Prepare data for correlation
            checkpoint_accs = [checkpoint_data[eid] for eid in common_ids]
            final_accs = [final_data_dict[eid] for eid in common_ids]
            
            # Calculate correlations
            try:
                pearson_corr, pearson_p = pearsonr(checkpoint_accs, final_accs)
                spearman_corr, spearman_p = spearmanr(checkpoint_accs, final_accs)
            except:
                continue
            
            all_results.append({
                'checkpoint_epoch': checkpoint_epoch,
                'final_epoch': final_epoch,
                'checkpoint_size': checkpoint_size,
                'n_samples': len(common_ids),
                'pearson_correlation': pearson_corr,
                'pearson_pvalue': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_pvalue': spearman_p,
                'mean_checkpoint_acc': np.mean(checkpoint_accs),
                'std_checkpoint_acc': np.std(checkpoint_accs),
                'mean_final_acc': np.mean(final_accs),
                'std_final_acc': np.std(final_accs),
                'mean_accuracy_diff': np.mean(final_accs) - np.mean(checkpoint_accs)
            })
    
    if not all_results:
        print("[ERROR] No valid correlation results computed!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Save comprehensive results
    output_file = output_dir / f'checkpoint_correlation_all_to_epoch_{final_epoch}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✓ Comprehensive results saved to: {output_file}")
    
    # Create visualizations
    create_all_epochs_visualizations(results_df, output_dir, final_epoch, available_epochs, checkpoint_sizes)
    
    # Create summary by checkpoint epoch
    print("\n" + "="*80)
    print(f"SUMMARY BY CHECKPOINT EPOCH (all correlated to Final Epoch {final_epoch})")
    print("="*80)
    
    for checkpoint_epoch in available_epochs:
        epoch_results = results_df[results_df['checkpoint_epoch'] == checkpoint_epoch]
        if len(epoch_results) == 0:
            continue
        
        print(f"\nCheckpoint Epoch {checkpoint_epoch} → Final Epoch {final_epoch}:")
        
        # Find best checkpoint size for this epoch
        best_idx = epoch_results['pearson_correlation'].idxmax()
        best = epoch_results.loc[best_idx]
        
        print(f"  Best checkpoint size: {int(best['checkpoint_size']):,} samples")
        print(f"    Pearson r: {best['pearson_correlation']:.4f}, Spearman ρ: {best['spearman_correlation']:.4f}")
        print(f"    n={int(best['n_samples'])} candidates")
        
        # Show all checkpoint sizes for this epoch
        epoch_results_sorted = epoch_results.sort_values('checkpoint_size')
        print(f"  All checkpoint sizes:")
        for _, row in epoch_results_sorted.iterrows():
            print(f"    {int(row['checkpoint_size']):>6,} samples: r={row['pearson_correlation']:.3f}, ρ={row['spearman_correlation']:.3f}")
    
    return results_df


def _save_figure(fig, output_dir: Path, base_name: str):
    """Helper function to save a figure as both PNG and SVG."""
    output_file_png = output_dir / f'{base_name}.png'
    output_file_svg = output_dir / f'{base_name}.svg'
    fig.savefig(output_file_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_file_svg, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_file_png.name} and {output_file_svg.name}")
    plt.close(fig)


def create_all_epochs_visualizations(results_df: pd.DataFrame, output_dir: Path, final_epoch: int,
                                     available_epochs: list, checkpoint_sizes: list):
    """Create comprehensive visualizations for all epochs correlation analysis."""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Combine 16k and 32k checkpoints by averaging their correlations
    # Create a copy of results_df for processing
    vis_df = results_df.copy()
    
    # Define sizes to combine (16k and 32k)
    sizes_to_combine = [16392, 32768]
    combined_size = 24580  # Average of 16k and 32k for display
    has_combined_size = False
    
    # Check if both 16k and 32k exist in the data
    if all(size in checkpoint_sizes for size in sizes_to_combine):
        has_combined_size = True
        # For each epoch, combine 16k and 32k by averaging correlations
        combined_results = []
        for epoch in available_epochs:
            epoch_data = vis_df[vis_df['checkpoint_epoch'] == epoch]
            size_16k = epoch_data[epoch_data['checkpoint_size'] == 16392]
            size_32k = epoch_data[epoch_data['checkpoint_size'] == 32768]
            
            if len(size_16k) > 0 and len(size_32k) > 0:
                # Average the correlations
                combined_row = size_16k.iloc[0].copy()
                combined_row['checkpoint_size'] = combined_size
                combined_row['pearson_correlation'] = (size_16k.iloc[0]['pearson_correlation'] + 
                                                       size_32k.iloc[0]['pearson_correlation']) / 2
                combined_row['spearman_correlation'] = (size_16k.iloc[0]['spearman_correlation'] + 
                                                        size_32k.iloc[0]['spearman_correlation']) / 2
                combined_row['n_samples'] = min(size_16k.iloc[0]['n_samples'], size_32k.iloc[0]['n_samples'])
                combined_results.append(combined_row)
        
        # Remove original 16k and 32k entries and add combined ones
        vis_df = vis_df[~vis_df['checkpoint_size'].isin(sizes_to_combine)]
        if combined_results:
            vis_df = pd.concat([vis_df, pd.DataFrame(combined_results)], ignore_index=True)
        
        # Update checkpoint_sizes for visualization
        vis_checkpoint_sizes = [s for s in checkpoint_sizes if s not in sizes_to_combine] + [combined_size]
        vis_checkpoint_sizes = sorted(vis_checkpoint_sizes)
    else:
        vis_checkpoint_sizes = sorted(checkpoint_sizes)
    
    # Prepare data for heatmaps
    pearson_pivot = vis_df.pivot_table(
        index='checkpoint_epoch', 
        columns='checkpoint_size', 
        values='pearson_correlation'
    )
    spearman_pivot = vis_df.pivot_table(
        index='checkpoint_epoch', 
        columns='checkpoint_size', 
        values='spearman_correlation'
    )
    
    # Plot 1: Pearson Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pearson_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.8, 
                vmin=0, vmax=1, cbar_kws={'label': 'Pearson r'}, ax=ax)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Checkpoint Epoch', fontsize=18)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'01_pearson_correlation_heatmap_epoch_{final_epoch}')
    
    # Plot 2: Spearman Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(spearman_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.8,
                vmin=0, vmax=1, cbar_kws={'label': 'Spearman ρ'}, ax=ax)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Checkpoint Epoch', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'02_spearman_correlation_heatmap_epoch_{final_epoch}')
    
    # Plot 3: Correlation by Epoch (line plot for each checkpoint size) - BOTH Pearson and Spearman
    fig, ax = plt.subplots(figsize=(12, 8))
    for size in sorted(vis_checkpoint_sizes):
        size_data = vis_df[vis_df['checkpoint_size'] == size].sort_values('checkpoint_epoch')
        if len(size_data) > 0:
            # Format label for combined size
            if has_combined_size and size == combined_size:
                label_base = '16k-32k'
            else:
                label_base = f'{int(size):,}'
            
            ax.plot(size_data['checkpoint_epoch'], size_data['pearson_correlation'], 
                    'o-', label=f'{label_base} (Pearson)', linewidth=2, markersize=6, alpha=0.8)
            ax.plot(size_data['checkpoint_epoch'], size_data['spearman_correlation'], 
                    's--', label=f'{label_base} (Spearman)', linewidth=2, markersize=5, alpha=0.8)
    ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, label='Strong (0.8)')
    ax.set_xlabel('Checkpoint Epoch', fontsize=18)
    ax.set_ylabel('Correlation Coefficient', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(title='Checkpoint Size', fontsize=12, ncol=2, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'03_correlation_by_epoch_epoch_{final_epoch}')
    
    # Plot 4: Correlation by Checkpoint Size (line plot for each epoch) - BOTH Pearson and Spearman
    fig, ax = plt.subplots(figsize=(12, 8))
    for epoch in sorted(available_epochs):
        epoch_data = vis_df[vis_df['checkpoint_epoch'] == epoch].sort_values('checkpoint_size')
        if len(epoch_data) > 0:
            ax.plot(epoch_data['checkpoint_size'], epoch_data['pearson_correlation'], 
                    'o-', label=f'Epoch {epoch} (Pearson)', linewidth=2, markersize=6, alpha=0.8)
            ax.plot(epoch_data['checkpoint_size'], epoch_data['spearman_correlation'], 
                    's--', label=f'Epoch {epoch} (Spearman)', linewidth=2, markersize=5, alpha=0.8)
    ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, label='Strong (0.8)')
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Correlation Coefficient', fontsize=18)
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12, ncol=2, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'04_correlation_by_checkpoint_size_epoch_{final_epoch}')
    
    # Plot 5: Best correlation per epoch (both Pearson and Spearman)
    fig, ax = plt.subplots(figsize=(10, 8))
    best_pearson_per_epoch = vis_df.loc[vis_df.groupby('checkpoint_epoch')['pearson_correlation'].idxmax()]
    best_spearman_per_epoch = vis_df.loc[vis_df.groupby('checkpoint_epoch')['spearman_correlation'].idxmax()]
    best_pearson_per_epoch = best_pearson_per_epoch.sort_values('checkpoint_epoch')
    best_spearman_per_epoch = best_spearman_per_epoch.sort_values('checkpoint_epoch')
    
    x_pos = range(len(best_pearson_per_epoch))
    width = 0.35
    ax.bar([x - width/2 for x in x_pos], best_pearson_per_epoch['pearson_correlation'], 
            width, color='steelblue', alpha=0.7, label='Pearson')
    ax.bar([x + width/2 for x in x_pos], best_spearman_per_epoch['spearman_correlation'], 
            width, color='coral', alpha=0.7, label='Spearman')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong (0.8)')
    ax.set_xlabel('Checkpoint Epoch', fontsize=18)
    ax.set_ylabel('Best Correlation Coefficient', fontsize=18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Ep {int(e)}" for e in best_pearson_per_epoch['checkpoint_epoch']], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'05_best_correlation_per_epoch_epoch_{final_epoch}')
    
    # Plot 6: Optimal checkpoint size per epoch
    fig, ax = plt.subplots(figsize=(10, 8))
    best_per_epoch = vis_df.loc[vis_df.groupby('checkpoint_epoch')['pearson_correlation'].idxmax()]
    best_per_epoch = best_per_epoch.sort_values('checkpoint_epoch')
    ax.bar(range(len(best_per_epoch)), best_per_epoch['checkpoint_size'], 
            color='coral', alpha=0.7)
    ax.set_xlabel('Checkpoint Epoch', fontsize=18)
    ax.set_ylabel('Optimal Checkpoint Size (samples)', fontsize=18)
    ax.set_xticks(range(len(best_per_epoch)))
    ax.set_xticklabels([f"Ep {int(e)}" for e in best_per_epoch['checkpoint_epoch']], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'06_optimal_checkpoint_size_per_epoch_epoch_{final_epoch}')
    
    # Plot 7: Correlation difference (Pearson - Spearman)
    fig, ax = plt.subplots(figsize=(10, 8))
    vis_df['corr_diff'] = vis_df['pearson_correlation'] - vis_df['spearman_correlation']
    diff_pivot = vis_df.pivot_table(
        index='checkpoint_epoch', 
        columns='checkpoint_size', 
        values='corr_diff'
    )
    sns.heatmap(diff_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Pearson - Spearman'}, ax=ax)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Checkpoint Epoch', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'07_correlation_difference_epoch_{final_epoch}')
    
    # Plot 8: Sample size per combination
    fig, ax = plt.subplots(figsize=(10, 8))
    n_samples_pivot = vis_df.pivot_table(
        index='checkpoint_epoch', 
        columns='checkpoint_size', 
        values='n_samples'
    )
    sns.heatmap(n_samples_pivot, annot=True, fmt='.0f', cmap='Blues',
                cbar_kws={'label': 'Number of Samples'}, ax=ax)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Checkpoint Epoch', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'08_sample_size_per_combination_epoch_{final_epoch}')
    
    # Plot 9: Mean accuracy difference
    fig, ax = plt.subplots(figsize=(10, 8))
    acc_diff_pivot = vis_df.pivot_table(
        index='checkpoint_epoch', 
        columns='checkpoint_size', 
        values='mean_accuracy_diff'
    )
    sns.heatmap(acc_diff_pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Accuracy Difference (%)'}, ax=ax)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Checkpoint Epoch', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    _save_figure(fig, output_dir, f'09_mean_accuracy_difference_epoch_{final_epoch}')


def analyze_checkpoint_correlation(checkpoint_file: str, final_epoch: int = 1, checkpoint_epoch: int = 1, output_dir: Path = None):
    """
    Analyze correlation between different checkpoint sizes and final accuracy.
    
    Args:
        checkpoint_file: Path to checkpoint_validation.csv file
        final_epoch: Epoch to use as final accuracy benchmark (default: 1)
        checkpoint_epoch: Epoch to analyze checkpoints from (default: 1)
        output_dir: Directory to save results (default: parent of checkpoint_file)
    """
    if output_dir is None:
        output_dir = Path(checkpoint_file).parent / "checkpoint_correlation_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CHECKPOINT SIZE CORRELATION ANALYSIS")
    print(f"Checkpoints at Epoch {checkpoint_epoch} vs Final Accuracy at Epoch {final_epoch}")
    print("="*80)
    print(f"Reading data from: {checkpoint_file}\n")
    
    # Read the checkpoint validation data
    df = pd.read_csv(checkpoint_file)
    
    # Check if requested epochs exist
    available_epochs = sorted(df['epoch'].unique())
    print(f"Available epochs in data: {available_epochs}")
    
    if checkpoint_epoch not in available_epochs:
        print(f"\n[ERROR] Checkpoint epoch {checkpoint_epoch} not found in data!")
        print(f"Please choose from: {available_epochs}")
        sys.exit(1)
    
    if final_epoch not in available_epochs:
        print(f"\n[ERROR] Final epoch {final_epoch} not found in data!")
        print(f"Please choose from: {available_epochs}")
        sys.exit(1)
    
    # Filter for checkpoint epoch
    df_checkpoint = df[df['epoch'] == checkpoint_epoch].copy()
    
    # Get all unique eval_ids at checkpoint epoch
    all_ids = df_checkpoint['eval_id'].unique()
    print(f"Total candidates at epoch {checkpoint_epoch}: {len(all_ids)}")
    
    # Get unique checkpoint sizes at checkpoint epoch
    checkpoint_sizes = sorted(df_checkpoint['checkpoint_train_size'].unique())
    print(f"Checkpoint sizes available: {checkpoint_sizes}")
    
    # Determine the largest checkpoint size (this is our "final" reference)
    max_checkpoint_size = max(checkpoint_sizes)
    print(f"Using {max_checkpoint_size} samples at epoch {final_epoch} as final accuracy reference\n")
    
    # Get the final accuracy at final_epoch (using largest checkpoint size)
    df_final = df[df['epoch'] == final_epoch].copy()
    final_data = df_final[
        df_final['checkpoint_train_size'] == max_checkpoint_size
    ][['eval_id', 'val_acc']].rename(columns={'val_acc': 'final_acc'})
    
    # Find candidates that have the final checkpoint
    candidates_with_final = set(final_data['eval_id'].unique())
    print(f"Candidates with {max_checkpoint_size} samples checkpoint: {len(candidates_with_final)}")
    
    if len(candidates_with_final) == 0:
        print(f"\n[ERROR] No candidates found with {max_checkpoint_size} samples checkpoint!")
        return
    
    print(f"Common eval_ids: {sorted(candidates_with_final)}\n")
    
    # Calculate correlations for each checkpoint size
    results = []
    
    print("="*80)
    print("CORRELATION RESULTS BY CHECKPOINT SIZE")
    print(f"(Checkpoint Epoch {checkpoint_epoch} vs Final Epoch {final_epoch} at {max_checkpoint_size} samples)")
    print("="*80)
    
    for size in checkpoint_sizes:
        # Skip the final checkpoint size if comparing within same epoch (perfect correlation with itself)
        if size == max_checkpoint_size and checkpoint_epoch == final_epoch:
            continue
            
        # Get data for this checkpoint size at checkpoint_epoch
        checkpoint_data = df_checkpoint[
            (df_checkpoint['checkpoint_train_size'] == size) &
            (df_checkpoint['eval_id'].isin(candidates_with_final))
        ][['eval_id', 'val_acc']].rename(columns={'val_acc': 'checkpoint_acc'})
        
        # Merge with final accuracy
        merged = pd.merge(checkpoint_data, final_data, on='eval_id', how='inner')
        
        if len(merged) < 2:
            print(f"\nCheckpoint Size: {size} samples")
            print(f"  ⚠️  Insufficient data (n={len(merged)}) - skipping")
            continue
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(merged['checkpoint_acc'], merged['final_acc'])
        spearman_corr, spearman_p = spearmanr(merged['checkpoint_acc'], merged['final_acc'])
        
        # Calculate additional statistics
        accuracy_diff = merged['final_acc'].mean() - merged['checkpoint_acc'].mean()
        
        results.append({
            'checkpoint_size': size,
            'checkpoint_epoch': checkpoint_epoch,
            'final_epoch': final_epoch,
            'n_samples': len(merged),
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_p,
            'mean_checkpoint_acc': merged['checkpoint_acc'].mean(),
            'std_checkpoint_acc': merged['checkpoint_acc'].std(),
            'mean_final_acc': merged['final_acc'].mean(),
            'std_final_acc': merged['final_acc'].std(),
            'mean_accuracy_diff': accuracy_diff
        })
        
        # Determine significance
        pearson_sig = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else "ns"
        spearman_sig = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else "ns"
        
        print(f"\n{'='*80}")
        print(f"Checkpoint Size: {size} samples (Epoch {checkpoint_epoch})")
        print(f"{'='*80}")
        print(f"  Number of candidates: {len(merged)}")
        print(f"  Pearson Correlation:  r = {pearson_corr:7.4f}, p = {pearson_p:.4e} {pearson_sig}")
        print(f"  Spearman Correlation: ρ = {spearman_corr:7.4f}, p = {spearman_p:.4e} {spearman_sig}")
        print(f"  Checkpoint Acc (Epoch {checkpoint_epoch}, {size} samples):  {merged['checkpoint_acc'].mean():6.2f}% ± {merged['checkpoint_acc'].std():.2f}%")
        print(f"  Final Accuracy (Epoch {final_epoch}, {max_checkpoint_size} samples): {merged['final_acc'].mean():6.2f}% ± {merged['final_acc'].std():.2f}%")
        print(f"  Mean Accuracy Difference: {accuracy_diff:6.2f}%")
    
    if not results:
        print("\n[ERROR] No valid correlation results computed!")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / f'checkpoint_size_correlation_analysis_ep{checkpoint_epoch}_to_ep{final_epoch}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Create visualizations
    create_visualizations(results_df, output_dir, max_checkpoint_size, checkpoint_epoch, final_epoch)
    
    # Find best checkpoint sizes
    print("\n" + "="*80)
    print("SUMMARY: OPTIMAL CHECKPOINT SIZES")
    print(f"Checkpoint Epoch {checkpoint_epoch} → Final Epoch {final_epoch} (at {max_checkpoint_size} samples)")
    print("="*80)
    
    # Best by Pearson correlation
    best_pearson_idx = results_df['pearson_correlation'].idxmax()
    best_pearson = results_df.loc[best_pearson_idx]
    print(f"\n🏆 Best by Pearson Correlation:")
    print(f"   Checkpoint Size:  {int(best_pearson['checkpoint_size']):,} samples")
    print(f"   Pearson r:        {best_pearson['pearson_correlation']:.4f}")
    print(f"   Spearman ρ:       {best_pearson['spearman_correlation']:.4f}")
    print(f"   P-value:          {best_pearson['pearson_pvalue']:.4e}")
    print(f"   Candidates:       {int(best_pearson['n_samples'])}")
    
    # Best by Spearman correlation
    best_spearman_idx = results_df['spearman_correlation'].idxmax()
    best_spearman = results_df.loc[best_spearman_idx]
    print(f"\n🏆 Best by Spearman Correlation:")
    print(f"   Checkpoint Size:  {int(best_spearman['checkpoint_size']):,} samples")
    print(f"   Spearman ρ:       {best_spearman['spearman_correlation']:.4f}")
    print(f"   Pearson r:        {best_spearman['pearson_correlation']:.4f}")
    print(f"   P-value:          {best_spearman['spearman_pvalue']:.4e}")
    print(f"   Candidates:       {int(best_spearman['n_samples'])}")
    
    # Find checkpoint sizes with strong correlation (r or ρ > 0.8)
    strong_corr = results_df[
        (results_df['pearson_correlation'] > 0.8) | 
        (results_df['spearman_correlation'] > 0.8)
    ].sort_values('checkpoint_size')
    
    if not strong_corr.empty:
        print(f"\n📊 Checkpoint sizes with strong correlation (r or ρ > 0.8):")
        for _, row in strong_corr.iterrows():
            print(f"   {int(row['checkpoint_size']):>6,} samples: "
                  f"Pearson={row['pearson_correlation']:.3f}, "
                  f"Spearman={row['spearman_correlation']:.3f}")
        
        # Recommend smallest size with strong correlation
        smallest_strong = strong_corr.iloc[0]
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use checkpoint size: {int(smallest_strong['checkpoint_size']):,} samples")
        print(f"   This is the smallest size with strong predictive power,")
        print(f"   offering efficient training while maintaining high correlation.")
    else:
        print(f"\n⚠️  Note: No checkpoint sizes achieved strong correlation (>0.8)")
        print(f"   Consider using the best available checkpoint size.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def create_visualizations(results_df: pd.DataFrame, output_dir: Path, max_checkpoint_size: int, 
                         checkpoint_epoch: int, final_epoch: int):
    """Create visualization plots for correlation analysis."""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Pearson correlation vs checkpoint size
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(results_df['checkpoint_size'], results_df['pearson_correlation'], 
            'o-', linewidth=2, markersize=8, color='blue', label='Pearson r')
    ax.axhline(y=0.9, color='darkgreen', linestyle='--', alpha=0.5, label='Very strong (0.9)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong (0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Pearson Correlation (r)', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'01_pearson_correlation_ep{checkpoint_epoch}_to_ep{final_epoch}')
    
    # Plot 2: Spearman correlation vs checkpoint size
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(results_df['checkpoint_size'], results_df['spearman_correlation'], 
            'o-', linewidth=2, markersize=8, color='red', label='Spearman ρ')
    ax.axhline(y=0.9, color='darkgreen', linestyle='--', alpha=0.5, label='Very strong (0.9)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong (0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Spearman Correlation (ρ)', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'02_spearman_correlation_ep{checkpoint_epoch}_to_ep{final_epoch}')
    
    # Plot 3: Both correlations together
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(results_df['checkpoint_size'], results_df['pearson_correlation'], 
            'o-', linewidth=2, markersize=8, color='blue', label='Pearson r')
    ax.plot(results_df['checkpoint_size'], results_df['spearman_correlation'], 
            's-', linewidth=2, markersize=8, color='red', label='Spearman ρ')
    ax.axhline(y=0.9, color='darkgreen', linestyle='--', alpha=0.5)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Correlation Coefficient', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'03_pearson_vs_spearman_ep{checkpoint_epoch}_to_ep{final_epoch}')
    
    # Plot 4: Mean accuracy by checkpoint size
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(results_df['checkpoint_size'], results_df['mean_checkpoint_acc'], 
            'o-', linewidth=2, markersize=8, label=f'Checkpoint Acc (Epoch {checkpoint_epoch})', color='orange')
    ax.axhline(y=results_df['mean_final_acc'].iloc[0], 
               linewidth=2, linestyle='--', color='green', 
               label=f'Final Acc Epoch {final_epoch} ({max_checkpoint_size} samples)', alpha=0.7)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Mean Validation Accuracy (%)', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'04_accuracy_by_checkpoint_size_ep{checkpoint_epoch}_to_ep{final_epoch}')
    
    # Plot 5: Accuracy difference vs checkpoint size
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(results_df['checkpoint_size'], results_df['mean_accuracy_diff'], 
            'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel(f'Mean Accuracy Difference\n(Final {max_checkpoint_size} - Checkpoint) (%)', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'05_accuracy_difference_ep{checkpoint_epoch}_to_ep{final_epoch}')
    
    # Plot 6: Number of samples per checkpoint size
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(range(len(results_df)), results_df['n_samples'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Checkpoint Size (samples)', fontsize=18)
    ax.set_ylabel('Number of Candidates', fontsize=18)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels([f"{int(s):,}" for s in results_df['checkpoint_size']], 
                        rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_figure(fig, output_dir, f'06_sample_size_per_checkpoint_ep{checkpoint_epoch}_to_ep{final_epoch}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze correlation between checkpoint sizes and final accuracy across epochs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all epochs and checkpoints (comprehensive analysis)
  python scripts/analyze_checkpoint_correlation.py logs/correlation/MNIST/run_20251128-171506/checkpoint_validation.csv --all-epochs

  # Same, but use epoch 3 as final benchmark instead of max
  python scripts/analyze_checkpoint_correlation.py logs/correlation/MNIST/run_20251128-171506/checkpoint_validation.csv --all-epochs --final-epoch 3

  # Compare checkpoints at epoch 1 to final epoch 10 accuracy
  python scripts/analyze_checkpoint_correlation.py logs/correlation/MNIST/run_20251128-171506/checkpoint_validation.csv --checkpoint-epoch 1 --final-epoch 10
  
  # Compare checkpoints at epoch 3 to final epoch 5 accuracy
  python scripts/analyze_checkpoint_correlation.py logs/correlation/MNIST/run_20251128-171506/checkpoint_validation.csv --checkpoint-epoch 3 --final-epoch 5
        """
    )
    
    parser.add_argument('checkpoint_file', type=str,
                        help='Path to checkpoint_validation.csv file')
    parser.add_argument('--final-epoch', type=int, default=None,
                        help='Epoch to use as final accuracy benchmark (default: max available). Works with --all-epochs and single-epoch mode.')
    parser.add_argument('--checkpoint-epoch', type=int, default=None,
                        help='Epoch to analyze checkpoints from (default: 1)')
    parser.add_argument('--all-epochs', action='store_true',
                        help='Analyze correlations for all epochs and checkpoint sizes')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: checkpoint_correlation_analysis/ in run directory)')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint_file).exists():
        print(f"Error: File not found: {args.checkpoint_file}")
        sys.exit(1)
    
    checkpoint_path = Path(args.checkpoint_file)
    run_dir = checkpoint_path.parent
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir / "checkpoint_correlation_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all_epochs:
        # Comprehensive analysis of all epochs (--final-epoch optional)
        analyze_all_epochs_correlations(args.checkpoint_file, output_dir, final_epoch=args.final_epoch)
    else:
        # Single epoch analysis
        # Determine defaults
        df = pd.read_csv(args.checkpoint_file)
        available_epochs = sorted(df['epoch'].unique())
        
        final_epoch = args.final_epoch if args.final_epoch is not None else max(available_epochs)
        checkpoint_epoch = args.checkpoint_epoch if args.checkpoint_epoch is not None else 1
        
        analyze_checkpoint_correlation(args.checkpoint_file, final_epoch, checkpoint_epoch, output_dir)


if __name__ == "__main__":
    main()

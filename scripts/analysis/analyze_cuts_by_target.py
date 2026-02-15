#!/usr/bin/env python3
"""
Analyze how the number of cuts changes for all circuits when varying cut target qubits.

Usage:
    python analyze_cuts_by_target.py --csv_path logs/nsga-ii/MNIST/run_20260125-132710/nsga_evals.csv --cut_targets 2,3,4,5,6
"""

import argparse
import sys
import re
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qnas.quantum.metrics import _qlayer_to_wirecut_string


# Patterns for parsing circuit strings (extracted from cutter.py, no TensorFlow needed)
patterns = {
    'rot': (re.compile(r'r(y|z|x)\((-?\d+(\.\d+)?)\) q\[(\d+)\];'),
             'single qubit parametrized gate e.g. ry(0.1) q[0]', lambda m: int(m.group(4))),
    'cx': (re.compile(r'cx q\[(\d+)\],q\[(\d+)\];'),
           'CNOT gate e.g. cx q[0],q[1]',
           lambda m: (int(m.group(1)), int(m.group(2)))),
}


def cut_placement_standalone(input_str, target_qubits):
    """Standalone version of cut_placement that doesn't require TensorFlow."""
    temp_string = ""
    subwires = set()
    split_str = input_str.splitlines()
    subwires_wire_list = []
    for string in split_str:
        line = string.strip()
        for pattern, definition, args_func in patterns.values():
            match = pattern.match(line)
            if match:
                wire = args_func(match)
                if isinstance(wire, int):
                    wire = (wire,)
                else:
                    wire = tuple(wire)
                if len(subwires.union(wire)) <= target_qubits:
                    subwires.update(wire)
                    temp_string += string + '\n'
                else:
                    temp_string += "CUT HERE\n" + string + '\n'
                    subwires_wire_list.append(list(subwires))
                    subwires = set(wire)
    if subwires:
        subwires_wire_list.append(list(subwires))
    return temp_string.strip(), subwires_wire_list


def parse_cnot_modes(cnot_modes_str: str) -> list:
    """Parse CNOT modes from string like 'even-even' or 'all-odd'."""
    mode_map = {'all': 0, 'odd': 1, 'even': 2, 'none': 3}
    modes = cnot_modes_str.split('-')
    return [mode_map.get(m.lower(), 0) for m in modes]


def calculate_cuts_for_target(n_qubits: int, depth: int, ent_ranges: list, cnot_modes: list, 
                             cut_target_qubits: int) -> dict:
    """Calculate the number of cuts and subcircuits for a given circuit configuration."""
    if cut_target_qubits <= 0:
        return {'num_cuts': 0, 'num_subcircuits': 1}
    
    # Generate circuit string
    qtxt = _qlayer_to_wirecut_string(n_qubits, depth, ent_ranges, cnot_modes)
    
    # Apply cutting
    cut_circuit, subwires_list = cut_placement_standalone(qtxt, cut_target_qubits)
    
    # Count cuts (number of "CUT HERE" markers)
    num_cuts = cut_circuit.count("CUT HERE")
    num_subcircuits = len(subwires_list)
    
    return {
        'num_cuts': num_cuts,
        'num_subcircuits': num_subcircuits
    }


def analyze_all_circuits(csv_path: str, cut_targets: list, output_csv: str = None):
    """Analyze all circuits in the CSV for different cut target qubit values."""
    
    # If output_csv not specified, save in same directory as input CSV
    if output_csv is None:
        csv_path_obj = Path(csv_path)
        output_csv = str(csv_path_obj.parent / "cuts_analysis.csv")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Analyzing {len(df)} circuits from {csv_path}")
    print(f"Cut target qubits to test: {cut_targets}\n")
    
    # Prepare results
    results = []
    
    mode_map = {'all': 0, 'odd': 1, 'even': 2, 'none': 3}
    
    for idx, row in df.iterrows():
        eval_id = row['eval_id']
        n_qubits = int(row['n_qubits'])
        depth = int(row['depth'])
        ent_ranges = [int(x) for x in str(row['ent_ranges']).split('-')]
        cnot_modes_str = str(row['cnot_modes'])
        cnot_modes = [mode_map.get(m.lower(), 0) for m in cnot_modes_str.split('-')]
        
        result_row = {
            'eval_id': eval_id,
            'n_qubits': n_qubits,
            'depth': depth,
            'ent_ranges': '-'.join(map(str, ent_ranges)),
            'cnot_modes': cnot_modes_str
        }
        
        # Calculate cuts for each target
        for cut_target in cut_targets:
            try:
                cuts_result = calculate_cuts_for_target(n_qubits, depth, ent_ranges, cnot_modes, cut_target)
                result_row[f'cuts_target_{cut_target}'] = cuts_result['num_cuts']
                result_row[f'subcircuits_target_{cut_target}'] = cuts_result['num_subcircuits']
            except Exception as e:
                print(f"Error calculating cuts for {eval_id} with target {cut_target}: {e}")
                result_row[f'cuts_target_{cut_target}'] = None
                result_row[f'subcircuits_target_{cut_target}'] = None
        
        results.append(result_row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display summary
    print("\n" + "="*100)
    print("SUMMARY: Number of cuts by cut target qubits")
    print("="*100)
    
    # Create a summary table
    summary_cols = ['eval_id', 'n_qubits', 'depth'] + [f'cuts_target_{t}' for t in cut_targets]
    summary_df = results_df[summary_cols].copy()
    
    print("\nNumber of cuts:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("Number of subcircuits by cut target qubits")
    print("="*100)
    
    subcircuit_cols = ['eval_id', 'n_qubits', 'depth'] + [f'subcircuits_target_{t}' for t in cut_targets]
    subcircuit_df = results_df[subcircuit_cols].copy()
    
    print("\nNumber of subcircuits:")
    print(subcircuit_df.to_string(index=False))
    
    # Statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    for cut_target in cut_targets:
        cuts_col = f'cuts_target_{cut_target}'
        subcircuits_col = f'subcircuits_target_{cut_target}'
        if cuts_col in results_df.columns:
            avg_cuts = results_df[cuts_col].mean()
            max_cuts = results_df[cuts_col].max()
            min_cuts = results_df[cuts_col].min()
            avg_subcircuits = results_df[subcircuits_col].mean()
            print(f"\nCut target = {cut_target} qubits:")
            print(f"  Average cuts: {avg_cuts:.2f}")
            print(f"  Min cuts: {min_cuts}")
            print(f"  Max cuts: {max_cuts}")
            print(f"  Average subcircuits: {avg_subcircuits:.2f}")
    
    # Save to CSV if requested
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\nFull results saved to: {output_csv}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how number of cuts changes for all circuits when varying cut target qubits"
    )
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with circuit evaluations')
    parser.add_argument('--cut_targets', type=str, default='2,3,4,5,6',
                       help='Comma-separated list of cut target qubit values to test (default: 2,3,4,5,6)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Optional: Save full results to CSV file')
    
    args = parser.parse_args()
    
    # Parse cut targets
    cut_targets = [int(x.strip()) for x in args.cut_targets.split(',')]
    cut_targets = sorted(cut_targets)  # Sort for better presentation
    
    # Run analysis
    results_df = analyze_all_circuits(args.csv_path, cut_targets, args.output_csv)
    
    return results_df


if __name__ == "__main__":
    main()

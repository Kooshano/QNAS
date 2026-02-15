#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transpile the best circuit using Qiskit with fake IBM backends.

This script loads the best circuit configuration and transpiles it to
a fake IBM backend using qiskit_ibm_runtime.fake_provider.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from io import StringIO

# Prevent src modules from creating log directories when imported
os.environ["IMPORTED_AS_MODULE"] = "true"

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Qiskit Aer.*')
warnings.filterwarnings('ignore', message='.*nighthawk.*')

# Suppress Qiskit-specific warnings
os.environ['QISKIT_SUPPRESS_PACKAGING_WARNINGS'] = '1'

# Add project root directory to path to import from src
# Since this script is in scripts/utils/, we go up 3 levels
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Lazy import - will be imported when needed
QISKIT_AVAILABLE = None
_QiskitCircuit = None
_ClassicalRegister = None
_transpile = None
_fake_backends_module = None

def _import_qiskit():
    """Lazy import of Qiskit modules."""
    global QISKIT_AVAILABLE, _QiskitCircuit, _ClassicalRegister, _transpile, _fake_backends_module
    if QISKIT_AVAILABLE is not None:
        return QISKIT_AVAILABLE
    
    try:
        from qiskit import QuantumCircuit, transpile, ClassicalRegister
        _QiskitCircuit = QuantumCircuit
        _ClassicalRegister = ClassicalRegister
        _transpile = transpile
        
        # Try to import fake providers from qiskit-ibm-runtime
        try:
            import qiskit_ibm_runtime.fake_provider as fake_provider
            _fake_backends_module = fake_provider
            QISKIT_AVAILABLE = True
            return True
        except ImportError:
            # If qiskit-ibm-runtime is not available, provide helpful error
            QISKIT_AVAILABLE = False
            return False
        
    except ImportError as e:
        QISKIT_AVAILABLE = False
        return False

# Import from new modular structure
from qnas.utils.model_io import load_model_weights
from qnas.models.config import QConfig
from qnas.quantum.circuits import EMBED_TO_ROT, entangling_pairs, _filter_pairs_by_mode
import pennylane as qml
import numpy as np


def build_pennylane_circuit(cfg: QConfig):
    """
    Build the actual PennyLane circuit used in NSGA-II training.
    
    Parameters:
    -----------
    cfg : QConfig
        Circuit configuration
        
    Returns:
    --------
    qml.QNode
        PennyLane QNode representing the actual training circuit
    """
    # Create a device for visualization (doesn't need to match training device)
    dev = qml.device('default.qubit', wires=cfg.n_qubits)
    
    # Build the actual circuit structure used in training
    embed_kind = cfg.embed_kind.lower()
    
    def circuit(inputs, weights):
        if embed_kind in EMBED_TO_ROT:
            qml.AngleEmbedding(inputs, wires=range(cfg.n_qubits), rotation=EMBED_TO_ROT[embed_kind])
            if embed_kind == "angle-z":
                for q in range(cfg.n_qubits):
                    qml.Hadamard(wires=q)
        else:
            qml.AmplitudeEmbedding(inputs, wires=range(cfg.n_qubits), pad_with=0.0, normalize=False)
        
        for l in range(cfg.depth):
            # Use the same strongly_entangling_block structure as in training
            for q in range(cfg.n_qubits):
                rx, ry, rz = weights[l][q]
                qml.RX(rx, wires=q)
                qml.RY(ry, wires=q)
                qml.RZ(rz, wires=q)
            
            # Add entangling gates
            pairs = entangling_pairs(cfg.n_qubits, cfg.ent_ranges[l])
            pairs = _filter_pairs_by_mode(pairs, cfg.cnot_modes[l])
            for c, t in pairs:
                qml.CNOT(wires=[c, t])
        
        if embed_kind == "angle-z":
            return [qml.expval(qml.PauliX(i)) for i in range(cfg.n_qubits)]
        else:
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]
    
    qnode = qml.QNode(circuit, dev, interface="numpy")
    return qnode


def build_qiskit_circuit(cfg: QConfig):
    """
    Build a Qiskit circuit equivalent to the PennyLane circuit.
    
    Parameters:
    -----------
    cfg : QConfig
        Circuit configuration
        
    Returns:
    --------
    QuantumCircuit
        Qiskit circuit representation
    """
    if not _import_qiskit():
        raise ImportError("Qiskit is not available. Install with: pip install qiskit qiskit-ibm-runtime")
    
    qc = _QiskitCircuit(cfg.n_qubits)
    
    # Add embedding layer (for angle embeddings, we'll use parameterized gates)
    # Use small non-zero values to preserve circuit structure during transpilation
    if cfg.embed_kind in EMBED_TO_ROT:
        # For angle embeddings, we'll add parameterized rotation gates
        # These would normally be bound to input values during execution
        # Using 0.1 as placeholder to avoid being optimized away
        for q in range(cfg.n_qubits):
            if cfg.embed_kind == "angle-x":
                qc.rx(0.1, q)  # Placeholder - would be bound to input
            elif cfg.embed_kind == "angle-y":
                qc.ry(0.1, q)  # Placeholder - would be bound to input
            elif cfg.embed_kind == "angle-z":
                qc.rz(0.1, q)  # Placeholder - would be bound to input
                qc.h(q)  # Hadamard for angle-z as in PennyLane circuit
    else:
        # For amplitude embedding, we would use state preparation
        # This is more complex, so we'll just add a placeholder
        print(f"Note: Amplitude embedding requires state preparation (not fully implemented)")
    
    # Add variational layers (strongly entangling blocks)
    for l in range(cfg.depth):
        # Add rotation gates (RX, RY, RZ) for each qubit
        # Using small non-zero values to preserve structure
        for q in range(cfg.n_qubits):
            qc.rx(0.1, q)  # Placeholder - would be bound to weights
            qc.ry(0.1, q)  # Placeholder - would be bound to weights
            qc.rz(0.1, q)  # Placeholder - would be bound to weights
        
        # Add entangling gates (CNOTs)
        pairs = entangling_pairs(cfg.n_qubits, cfg.ent_ranges[l])
        pairs = _filter_pairs_by_mode(pairs, cfg.cnot_modes[l])
        for c, t in pairs:
            qc.cx(c, t)
    
    # Add measurements based on embedding type
    # Create classical register for measurements
    cr = _ClassicalRegister(cfg.n_qubits)
    qc.add_register(cr)
    
    # For angle-z, measure in X basis (already converted by Hadamard)
    # For other embeddings, measure in Z basis (standard)
    if cfg.embed_kind == "angle-z":
        # Already have Hadamard gates, so measure in X basis
        for q in range(cfg.n_qubits):
            qc.measure(q, cr[q])
    else:
        # Measure in Z basis (standard measurement)
        for q in range(cfg.n_qubits):
            qc.measure(q, cr[q])
    
    return qc


def select_fake_backend(n_qubits: int, backend_name: Optional[str] = None):
    """
    Select an appropriate fake backend with enough qubits.
    
    Parameters:
    -----------
    n_qubits : int
        Number of qubits required
    backend_name : str, optional
        Specific backend name to use (e.g., 'fake_athens', 'fake_london')
        If None, automatically selects the smallest suitable backend
        
    Returns:
    --------
    Backend
        Selected fake backend
    """
    if not _import_qiskit():
        raise ImportError("Qiskit is not available. Install with: pip install qiskit qiskit-ibm-runtime")
    
    try:
        # Get all available fake backends by inspecting the module
        # Suppress warnings during backend creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fake_backends = []
            backend_dict = {}  # Map backend names to backend objects
            for name in dir(_fake_backends_module):
                if name.startswith('Fake') and not name.startswith('_'):
                    try:
                        backend_class = getattr(_fake_backends_module, name)
                        backend = backend_class()
                        # Get num_qubits - handle both V1 and V2 backends
                        try:
                            num_qubits = backend.num_qubits
                        except AttributeError:
                            try:
                                num_qubits = len(backend.qubits) if hasattr(backend, 'qubits') else backend.configuration().n_qubits
                            except:
                                continue
                        fake_backends.append((backend, num_qubits))
                        # Store by backend name for lookup
                        try:
                            backend_dict[backend.name] = backend
                        except AttributeError:
                            pass
                    except Exception:
                        continue
        
        if not fake_backends:
            raise ValueError("No fake backends found")
        
        # If a specific backend name is provided, try to use it
        if backend_name:
            if backend_name in backend_dict:
                backend = backend_dict[backend_name]
                try:
                    backend_qubits = backend.num_qubits
                except AttributeError:
                    try:
                        backend_qubits = len(backend.qubits) if hasattr(backend, 'qubits') else backend.configuration().n_qubits
                    except:
                        backend_qubits = "Unknown"
                
                if isinstance(backend_qubits, int) and backend_qubits < n_qubits:
                    print(f"⚠ Warning: {backend_name} has only {backend_qubits} qubits, but circuit requires {n_qubits}")
                    print(f"  Using {backend_name} anyway - circuit may be truncated")
                else:
                    print(f"✓ Using specified backend: {backend_name} ({backend_qubits} qubits)")
                return backend
            else:
                print(f"⚠ Warning: Backend '{backend_name}' not found")
                print(f"  Available backends: {', '.join(sorted(backend_dict.keys()))}")
                print(f"  Falling back to automatic selection...")
        
        # Select backend with enough qubits (prefer smallest that fits)
        suitable_backends = [(b, nq) for b, nq in fake_backends if nq >= n_qubits]
        if suitable_backends:
            # Sort by number of qubits and select the smallest suitable one
            suitable_backends.sort(key=lambda x: x[1])
            return suitable_backends[0][0]
        else:
            # Fallback to largest available backend
            fake_backends.sort(key=lambda x: x[1], reverse=True)
            backend, num_qubits = fake_backends[0]
            print(f"⚠ Warning: No fake backend found with {n_qubits} qubits")
            print(f"  Using {backend.name} ({num_qubits} qubits) - circuit may be truncated")
            return backend
    except Exception as e:
        print(f"⚠ Warning: Error selecting fake backend: {e}")
        # Try to use a common small backend as fallback
        try:
            from qiskit_ibm_runtime.fake_provider import FakeAthensV2
            backend = FakeAthensV2()
            print(f"  Using fallback {backend.name} ({backend.num_qubits} qubits)")
            return backend
        except Exception:
            raise RuntimeError(f"Could not select fake backend: {e}")


def transpile_circuit(cfg: QConfig, optimization_level: int = 3, seed: int = 42, save_log: bool = True, backend_name: Optional[str] = None, run_folder: Optional[str] = None, weights_name: Optional[str] = None):
    """
    Transpile a circuit using Qiskit with a fake IBM backend.
    
    Parameters:
    -----------
    cfg : QConfig
        Circuit configuration
    optimization_level : int
        Transpilation optimization level (0-3)
    seed : int
        Random seed for transpilation
    save_log : bool
        Whether to save a log file
    backend_name : Optional[str]
        Name of the backend to use
    run_folder : Optional[str]
        Run folder name (e.g., "run_20251227-003334") to organize output
    weights_name : Optional[str]
        Name of the weights file (without extension) to create a subfolder for outputs
        
    Returns:
    --------
    QuantumCircuit
        Transpiled circuit, or None if error
    """
    if not _import_qiskit():
        print("✗ Error: Qiskit is not available")
        print("  Install with: pip install qiskit qiskit-ibm-runtime")
        return None
    
    try:
        print("\n== Transpiling circuit with Qiskit fake backend ==")
        print(f"Circuit config: {cfg.embed_kind}, {cfg.n_qubits} qubits, depth {cfg.depth}")
        print(f"  Entanglement ranges: {cfg.ent_ranges}")
        print(f"  CNOT modes: {cfg.cnot_modes}")
        
        # Build Qiskit circuit
        qc = build_qiskit_circuit(cfg)
        
        # Build the actual PennyLane circuit used in NSGA-II training
        print(f"\nBuilding original PennyLane circuit (from NSGA-II training)...")
        pennylane_qnode = build_pennylane_circuit(cfg)
        
        # Create dummy inputs/weights for visualization
        dummy_inputs = np.random.randn(cfg.n_qubits)
        dummy_weights = np.random.randn(cfg.depth, cfg.n_qubits, 3)
        
        # Draw the original PennyLane circuit
        print(f"\nOriginal PennyLane circuit (from NSGA-II training):")
        pennylane_draw = None
        pennylane_fig = None
        try:
            # Get text representation for console/log
            pennylane_draw = qml.draw(pennylane_qnode)(dummy_inputs, dummy_weights)
            print(pennylane_draw)
            # Get matplotlib figure for SVG/PNG export
            # qml.draw_mpl() returns (fig, ax) tuple, we need the figure
            draw_result = qml.draw_mpl(pennylane_qnode)(dummy_inputs, dummy_weights)
            if isinstance(draw_result, tuple) and len(draw_result) >= 1:
                pennylane_fig = draw_result[0]  # Extract figure from (fig, ax) tuple
            elif hasattr(draw_result, 'savefig'):
                pennylane_fig = draw_result  # It's already a figure
            else:
                print(f"  ⚠ Warning: Unexpected return type from qml.draw_mpl(): {type(draw_result)}")
                pennylane_fig = None
        except Exception as e:
            print(f"Could not display PennyLane circuit: {e}")
            import traceback
            traceback.print_exc()
            pennylane_draw = f"Error generating PennyLane circuit diagram: {e}"
            pennylane_fig = None
        
        # Now build Qiskit circuit for comparison and transpilation
        print(f"\nBuilding Qiskit circuit for transpilation...")
        
        print(f"\nQiskit circuit (before transpilation):")
        print(f"  Depth: {qc.depth()}")
        print(f"  Gates: {qc.size()}")
        print(f"  CNOTs: {qc.count_ops().get('cx', 0)}")
        
        # Show gate counts for Qiskit circuit
        print(f"\nQiskit circuit gate counts:")
        qiskit_gate_counts = qc.count_ops()
        for gate, count in sorted(qiskit_gate_counts.items()):
            print(f"  {gate}: {count}")
        
        # Draw the Qiskit circuit BEFORE transpilation
        print(f"\nQiskit circuit diagram (before transpilation):")
        try:
            qiskit_circuit_draw = qc.draw(output='text', fold=-1)
            print(qiskit_circuit_draw)
        except Exception as e:
            try:
                qiskit_circuit_str = str(qc)
                print(qiskit_circuit_str)
            except Exception:
                print(f"Could not display Qiskit circuit: {e}")
        
        # Select fake backend
        fake_backend = select_fake_backend(cfg.n_qubits, backend_name)
        print(f"\nUsing fake backend: {fake_backend.name}")
        print(f"  Number of qubits: {fake_backend.num_qubits}")
        print(f"  Coupling map: {fake_backend.coupling_map}")
        
        # Transpile the circuit (measurements are preserved)
        print(f"\nTranspiling circuit (optimization level {optimization_level})...")
        transpiled_qc = _transpile(
            qc,
            backend=fake_backend,
            optimization_level=optimization_level,
            seed_transpiler=seed
        )
        
        print(f"\nTranspiled circuit:")
        print(f"  Depth: {transpiled_qc.depth()}")
        print(f"  Gates: {transpiled_qc.size()}")
        print(f"  CNOTs: {transpiled_qc.count_ops().get('cx', 0)}")
        
        # Calculate improvement metrics
        original_cnots = qc.count_ops().get('cx', 0)
        transpiled_cnots = transpiled_qc.count_ops().get('cx', 0)
        original_depth = qc.depth()
        transpiled_depth = transpiled_qc.depth()
        
        print(f"\nTranspilation metrics:")
        print(f"  CNOT count: {original_cnots} → {transpiled_cnots} "
              f"({transpiled_cnots - original_cnots:+d})")
        print(f"  Circuit depth: {original_depth} → {transpiled_depth} "
              f"({transpiled_depth - original_depth:+d})")
        
        # Show gate counts for transpiled circuit
        print(f"\nTranspiled circuit gate counts:")
        gate_counts = transpiled_qc.count_ops()
        for gate, count in sorted(gate_counts.items()):
            print(f"  {gate}: {count}")
        
        # Draw the complete transpiled circuit
        print(f"\nTranspiled circuit:")
        try:
            # Get full text representation of the circuit
            circuit_draw = transpiled_qc.draw(output='text', fold=-1)
            print(circuit_draw)
        except Exception as e:
            # Fallback to string representation
            try:
                circuit_str = str(transpiled_qc)
                print(circuit_str)
            except Exception:
                print(f"Could not display circuit: {e}")
        
        # Generate and save graphical representation
        print(f"\nGenerating circuit diagrams...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            from matplotlib import pyplot as plt
            
            # Create output directory if it doesn't exist
            # Structure: outputs/circuit_diagrams/{weights_name}/ or outputs/{run_folder}/circuit_diagrams/{weights_name}/
            if run_folder:
                base_dir = Path("outputs") / run_folder / "circuit_diagrams"
            else:
                base_dir = Path("outputs/circuit_diagrams")
            
            # Create subfolder for the weights being used
            if weights_name:
                output_dir = base_dir / weights_name
            else:
                output_dir = base_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename base
            filename_base = f"{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}"
            
            # Save ORIGINAL PennyLane circuit as SVG and PNG
            if pennylane_fig is not None:
                original_pennylane_svg_path = output_dir / f"original_pennylane_{filename_base}.svg"
                try:
                    pennylane_fig.savefig(str(original_pennylane_svg_path), format='svg', bbox_inches='tight', facecolor='white')
                    print(f"  ✓ Saved original PennyLane SVG: {original_pennylane_svg_path}")
                except Exception as e:
                    print(f"  ⚠ Could not save PennyLane SVG: {e}")
                
                original_pennylane_png_path = output_dir / f"original_pennylane_{filename_base}.png"
                try:
                    pennylane_fig.savefig(str(original_pennylane_png_path), dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"  ✓ Saved original PennyLane PNG: {original_pennylane_png_path}")
                except Exception as e:
                    print(f"  ⚠ Could not save PennyLane PNG: {e}")
                
                plt.close(pennylane_fig)
            else:
                print(f"  ⚠ Could not save PennyLane SVG/PNG: No figure available")
            
            # Save PennyLane circuit as text file for reference
            original_pennylane_txt_path = output_dir / f"original_pennylane_{filename_base}.txt"
            try:
                if pennylane_draw:
                    with open(original_pennylane_txt_path, 'w') as f:
                        f.write("Original PennyLane Circuit (from NSGA-II training):\n")
                        f.write("=" * 70 + "\n")
                        f.write(str(pennylane_draw))
                    print(f"  ✓ Saved original PennyLane circuit text: {original_pennylane_txt_path}")
            except Exception as e:
                print(f"  ⚠ Could not save PennyLane text: {e}")
            
            # Save Qiskit circuit (before transpilation) as PNG for comparison
            qiskit_original_png_path = output_dir / f"original_qiskit_{filename_base}.png"
            try:
                fig_qiskit = qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'}, scale=0.8, fold=-1)
                fig_qiskit.savefig(str(qiskit_original_png_path), dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig_qiskit)
                print(f"  ✓ Saved original Qiskit PNG: {qiskit_original_png_path}")
            except Exception as e:
                print(f"  ⚠ Could not save original Qiskit PNG: {e}")
            
            # Save Qiskit circuit as SVG
            try:
                qiskit_original_svg_path = output_dir / f"original_qiskit_{filename_base}.svg"
                fig_qiskit = qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'}, scale=0.8, fold=-1)
                fig_qiskit.savefig(str(qiskit_original_svg_path), format='svg', bbox_inches='tight', facecolor='white')
                plt.close(fig_qiskit)
                print(f"  ✓ Saved original Qiskit SVG: {qiskit_original_svg_path}")
            except Exception:
                pass
            
            # Save TRANSPILED circuit as PNG
            transpiled_png_path = output_dir / f"transpiled_{filename_base}.png"
            try:
                fig = transpiled_qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'}, scale=0.8, fold=-1)
                fig.savefig(str(transpiled_png_path), dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"  ✓ Saved transpiled PNG: {transpiled_png_path}")
            except Exception as e:
                print(f"  ⚠ Could not save transpiled PNG: {e}")
            
            # Save TRANSPILED circuit as SVG
            try:
                transpiled_svg_path = output_dir / f"transpiled_{filename_base}.svg"
                fig = transpiled_qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'}, scale=0.8, fold=-1)
                fig.savefig(str(transpiled_svg_path), format='svg', bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"  ✓ Saved transpiled SVG: {transpiled_svg_path}")
            except Exception:
                pass
                
        except ImportError:
            print("  ⚠ Matplotlib not available for graphical output")
            print("    Install with: pip install matplotlib")
        except Exception as e:
            print(f"  ⚠ Could not generate circuit diagram: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✓ Transpilation completed successfully")
        
        # Save log file if requested
        if save_log:
            try:
                # Create output directory if it doesn't exist (use same dir as circuit diagrams)
                if run_folder:
                    base_log_dir = Path("outputs") / run_folder / "circuit_diagrams"
                else:
                    base_log_dir = Path("outputs/circuit_diagrams")
                
                # Create subfolder for the weights being used
                if weights_name:
                    log_output_dir = base_log_dir / weights_name
                else:
                    log_output_dir = base_log_dir
                log_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate log filename based on circuit config
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                log_filename = f"transpile_log_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}_{timestamp}.txt"
                log_path = log_output_dir / log_filename
                
                # Create summary log
                with open(log_path, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("Circuit Transpilation Log\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Circuit config: {cfg.embed_kind}, {cfg.n_qubits} qubits, depth {cfg.depth}\n")
                    f.write(f"  Entanglement ranges: {cfg.ent_ranges}\n")
                    f.write(f"  CNOT modes: {cfg.cnot_modes}\n")
                    f.write(f"  Learning rate: {cfg.learning_rate}\n")
                    f.write(f"  Shots: {cfg.shots}\n\n")
                    
                    f.write("Original PennyLane Circuit (from NSGA-II training):\n")
                    f.write("-" * 70 + "\n")
                    if pennylane_draw:
                        f.write(str(pennylane_draw))
                    else:
                        f.write("Could not generate PennyLane circuit diagram\n")
                    f.write("\n" + "-" * 70 + "\n\n")
                    
                    f.write("Qiskit Circuit (before transpilation):\n")
                    f.write(f"  Depth: {qc.depth()}\n")
                    f.write(f"  Gates: {qc.size()}\n")
                    f.write(f"  CNOTs: {qc.count_ops().get('cx', 0)}\n\n")
                    
                    f.write("Qiskit circuit gate counts:\n")
                    qiskit_gate_counts = qc.count_ops()
                    for gate, count in sorted(qiskit_gate_counts.items()):
                        f.write(f"  {gate}: {count}\n")
                    f.write("\n")
                    
                    f.write("Qiskit circuit diagram (before transpilation):\n")
                    f.write("-" * 70 + "\n")
                    try:
                        qiskit_circuit_draw = qc.draw(output='text', fold=-1)
                        f.write(qiskit_circuit_draw)
                    except Exception:
                        f.write(str(qc))
                    f.write("\n" + "-" * 70 + "\n\n")
                    
                    f.write(f"Backend: {fake_backend.name}\n")
                    f.write(f"  Number of qubits: {fake_backend.num_qubits}\n")
                    f.write(f"  Coupling map: {fake_backend.coupling_map}\n\n")
                    
                    f.write("Transpiled circuit:\n")
                    f.write(f"  Depth: {transpiled_qc.depth()}\n")
                    f.write(f"  Gates: {transpiled_qc.size()}\n")
                    f.write(f"  CNOTs: {transpiled_qc.count_ops().get('cx', 0)}\n\n")
                    
                    f.write("Transpilation metrics:\n")
                    f.write(f"  CNOT count: {original_cnots} → {transpiled_cnots} "
                           f"({transpiled_cnots - original_cnots:+d})\n")
                    f.write(f"  Circuit depth: {original_depth} → {transpiled_depth} "
                           f"({transpiled_depth - original_depth:+d})\n\n")
                    
                    f.write("Transpiled circuit gate counts:\n")
                    gate_counts = transpiled_qc.count_ops()
                    for gate, count in sorted(gate_counts.items()):
                        f.write(f"  {gate}: {count}\n")
                    f.write("\n")
                    
                    f.write("Transpiled circuit diagram:\n")
                    f.write("-" * 70 + "\n")
                    try:
                        circuit_draw = transpiled_qc.draw(output='text', fold=-1)
                        f.write(circuit_draw)
                    except Exception:
                        f.write(str(transpiled_qc))
                    f.write("\n" + "-" * 70 + "\n\n")
                    
                    f.write("Files generated:\n")
                    if run_folder:
                        base_path = f"outputs/{run_folder}/circuit_diagrams"
                    else:
                        base_path = "outputs/circuit_diagrams"
                    if weights_name:
                        base_path = f"{base_path}/{weights_name}"
                    f.write(f"  Original PennyLane SVG: {base_path}/original_pennylane_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.svg\n")
                    f.write(f"  Original PennyLane PNG: {base_path}/original_pennylane_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.png\n")
                    f.write(f"  Original PennyLane text: {base_path}/original_pennylane_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.txt\n")
                    f.write(f"  Original Qiskit PNG: {base_path}/original_qiskit_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.png\n")
                    f.write(f"  Original Qiskit SVG: {base_path}/original_qiskit_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.svg\n")
                    f.write(f"  Transpiled PNG: {base_path}/transpiled_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.png\n")
                    f.write(f"  Transpiled SVG: {base_path}/transpiled_{cfg.embed_kind}_nq{cfg.n_qubits}_d{cfg.depth}.svg\n")
                    f.write(f"  Log: {log_path}\n")
                
                print(f"  ✓ Saved log: {log_path}")
            except Exception as e:
                print(f"  ⚠ Could not save log file: {e}")
        
        return transpiled_qc
        
    except Exception as e:
        print(f"\n✗ Error during transpilation: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_config_from_weights(weights_path: Path) -> Optional[QConfig]:
    """Load circuit configuration from weights file."""
    try:
        state_dict, config_dict, metadata_dict = load_model_weights(str(weights_path))
        if config_dict is not None:
            return QConfig(
                embed_kind=config_dict['embed_kind'],
                n_qubits=config_dict['n_qubits'],
                depth=config_dict['depth'],
                ent_ranges=config_dict['ent_ranges'],
                cnot_modes=config_dict['cnot_modes'],
                learning_rate=config_dict['learning_rate'],
                shots=config_dict['shots']
            )
        return None
    except Exception as e:
        print(f"Error loading config from weights: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Transpile the best circuit using Qiskit fake backends"
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to weights file (.pt) containing circuit configuration"
    )
    parser.add_argument(
        "--embed",
        type=str,
        help="Embedding kind (angle-x, angle-y, angle-z, amplitude)"
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        help="Number of qubits"
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Circuit depth"
    )
    parser.add_argument(
        "--ent-ranges",
        type=str,
        help="Entanglement ranges (comma-separated, e.g., '1,2,3')"
    )
    parser.add_argument(
        "--cnot-modes",
        type=str,
        help="CNOT modes (comma-separated, e.g., '0,1,2')"
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Transpilation optimization level (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for transpilation (default: 42)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Specific fake backend name to use (e.g., 'fake_athens', 'fake_london'). "
             "Run 'python scripts/utils/list_fake_backends.py' to see all available backends. "
             "If not specified, automatically selects the smallest suitable backend."
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default=None,
        help="Run folder name (e.g., 'run_20251227-003334') to organize weights and outputs. "
             "Weights will be looked for in weights/{run-folder}/ and outputs saved to outputs/{run-folder}/"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = None
    weights_name = None  # Track the weights file name for output folder
    
    # Determine weights directory based on run folder
    if args.run_folder:
        weights_dir = Path("weights") / args.run_folder
    else:
        weights_dir = Path("weights")
    
    if args.weights:
        # Load from weights file
        weights_path = Path(args.weights)
        # If path is relative and run_folder is specified, try in run folder first
        if not weights_path.is_absolute() and args.run_folder:
            run_folder_path = weights_dir / weights_path.name
            if run_folder_path.exists():
                weights_path = run_folder_path
        if not weights_path.exists():
            print(f"✗ Error: Weights file not found: {weights_path}")
            return 1
        # Extract weights name (without extension) for output folder
        weights_name = weights_path.stem
        cfg = load_config_from_weights(weights_path)
        if cfg is None:
            print("✗ Error: Could not load configuration from weights file")
            return 1
    elif all([args.embed, args.n_qubits is not None, args.depth is not None,
              args.ent_ranges, args.cnot_modes]):
        # Build from command line arguments
        try:
            ent_ranges = [int(x.strip()) for x in args.ent_ranges.split(',')]
            cnot_modes = [int(x.strip()) for x in args.cnot_modes.split(',')]
            cfg = QConfig(
                embed_kind=args.embed,
                n_qubits=args.n_qubits,
                depth=args.depth,
                ent_ranges=ent_ranges[:args.depth],  # Truncate to depth
                cnot_modes=cnot_modes[:args.depth],  # Truncate to depth
                learning_rate=0.01,  # Default, not used for transpilation
                shots=0  # Default, not used for transpilation
            )
        except Exception as e:
            print(f"✗ Error parsing arguments: {e}")
            return 1
    else:
        # Try to find latest weights file in the appropriate directory
        if weights_dir.exists():
            weight_files = sorted(weights_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if weight_files:
                print(f"Loading configuration from latest weights file: {weight_files[0]}")
                # Extract weights name (without extension) for output folder
                weights_name = weight_files[0].stem
                cfg = load_config_from_weights(weight_files[0])
                if cfg is None:
                    print("✗ Error: Could not load configuration from weights file")
                    return 1
            else:
                print(f"✗ Error: No weights files found in {weights_dir} and no configuration provided")
                parser.print_help()
                return 1
        else:
            print(f"✗ Error: No weights directory found at {weights_dir} and no configuration provided")
            parser.print_help()
            return 1
    
    # Transpile the circuit
    transpiled_qc = transpile_circuit(cfg, args.optimization_level, args.seed, backend_name=args.backend, run_folder=args.run_folder, weights_name=weights_name)
    
    if transpiled_qc is None:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


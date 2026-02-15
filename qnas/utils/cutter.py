# cutter.py — warning-safe TensorFlow + PennyLane helpers
# Notes:
# - Silences absl/TF C++ logs during import (those "Unable to register cuDNN factory..." lines).
# - Keeps TensorFlow on CPU only (does NOT change CUDA_VISIBLE_DEVICES, so PyTorch can still use GPU).
# - Uses only tf.keras (no separate tf_keras wheel import).

import os, sys, contextlib, re

# Keep legacy tf.keras API (if you rely on it)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Tone down TF/absl log verbosity
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide INFO/WARN/ERROR; leave only FATAL
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_SEVERITY", "FATAL")

# --- targeted stderr suppression for noisy TF import ---
@contextlib.contextmanager
def _suppress_stderr():
    try:
        fd = sys.stderr.fileno()
    except Exception:
        # Python-level fallback
        old = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            yield
        finally:
            try:
                sys.stderr.close()
            except Exception:
                pass
            sys.stderr = old
        return
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        try:
            os.dup2(saved, fd)
        finally:
            os.close(saved)
            os.close(devnull)

import pennylane as qml
import numpy as np

# Import TensorFlow with stderr silenced to hide absl pre-init spam
with _suppress_stderr():
    import tensorflow as tf

# Keep TF on CPU (so GPU plugins are not touched by TF, avoiding more logs)
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# Quiet Python-level TF logging as well
try:
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
except Exception:
    pass

from tensorflow import keras
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float32')

patterns = {
    'rot': (re.compile(r'r(y|z|x)\((-?\d+(\.\d+)?)\) q\[(\d+)\];'),
             'single qubit parametrized gate e.g. ry(0.1) q[0]' ,lambda m: int(m.group(4))),
    'cx': (re.compile(r'cx q\[(\d+)\],q\[(\d+)\];'),
           'CNOT gate e.g. cx q[0],q[1]',
           lambda m: (int(m.group(1)), int(m.group(2)))),
}

func_map = {
    'ry': qml.RY,
    'rx': qml.RX,
    'rz': qml.RZ,
    'cx': qml.CNOT,
    'h': qml.Hadamard,
}


def string_sort(input_str, num_layers):
    temp_string = ""
    ignored_string = ""
    split_str = input_str.splitlines()
    counters = 0

    for i in range(num_layers):
        for string in split_str:
            line = string.strip()
            matched=False
            for key, (pattern, desc, args_func) in patterns.items():
                match = pattern.match(line)
                if match:
                    wire = args_func(match)
                    matched=True
                    if isinstance(wire, tuple):  # multi-qubit gate (e.g., cx)
                        wire_list = wire
                        position_not_found = True
                        reverse_string = ""
                        for gate in reversed(temp_string.split("\n")):
                            gate = gate.strip()
                            if gate:
                                for p, desc, af in patterns.values():
                                    gate_match = p.match(gate)
                                    if gate_match:
                                        gate_wire = af(gate_match)
                                        if not isinstance(gate_wire, tuple):
                                            gate_wire = (gate_wire,)
                                        if any(w in wire_list for w in gate_wire) and position_not_found:
                                            reverse_string += line + "\n" + gate + "\n"
                                            position_not_found = False
                                        else:
                                            reverse_string += gate + "\n"
                                        break
                                else:
                                    reverse_string += gate + "\n"
                        if position_not_found:
                            reverse_string += line + "\n"
                        temp_string = "\n".join(reversed(reverse_string.split("\n"))).strip() + "\n"
                        break
                    elif 'r' in line:
                        line = f"r{match.group(1)}({counters}) q[{match.group(4)}];"
                        counters += 1
                        temp_string += line + "\n"
                    else:
                        temp_string += string + '\n'
                    break
            if not matched:
                if line not in ignored_string.split("\n"):
                    ignored_string += line + '\n'
    return temp_string.strip(), ignored_string.strip()


def cut_placement(input_str, target_qubits):
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


def build_sub(input_str, subwires, measure_list):
    target_qubits = len(subwires)
    dev = qml.device('default.qubit', wires=target_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(features=inputs, wires=range(target_qubits), rotation='Y')
        for line in input_str.splitlines():
            line = line.strip()
            for key, (pattern, desc, args_func) in patterns.items():
                match = pattern.match(line)
                if match:
                    args = args_func(match)
                    if isinstance(args, int) and 'r' in line:
                        func_map[f"r{match.group(1)}"](weights[int(match.group(2))],
                                                       subwires.index(args))
                    elif not isinstance(args, int):
                        wires = [subwires.index(x) for x in args]
                        func_map[key](wires)
                    else:
                        func_map[match.group(1)](subwires.index(args))

        return [qml.expval(qml.PauliZ(j)) for j in measure_list]

    return circuit


def build_model(input_shape, classical_before, neurons_before, activation_before, classical_after, neurons_after,
                activation_after, qlayer_string, subwires, weight_shapes, weights, num_params, measure_list):
    unique_sets = set(element for sublist in subwires for element in sublist)
    oldqubits = len(unique_sets)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for i in range(classical_before):
        layer = tf.keras.layers.Dense(neurons_before[i], activation=activation_before[i])
        x = layer(x)

    states = tf.split(x, num_or_size_splits=oldqubits, axis=1)
    states_list = list(states)

    subcircuits = qlayer_string.split("CUT HERE")

    wire_list_ctr = 0
    draw_weights = [x / 10.0 for x in range(num_params)]
    draw_input = [x / 10.0 for x in range(oldqubits)]

    for str in subcircuits:
        subwire_list = subwires[wire_list_ctr]  # wires of that subcircuit
        target_qubits = len(subwire_list)

        substate_ctr = 0
        need_states = [states_list[i] for i in subwire_list]
        need_draw_input = [draw_input[i] for i in subwire_list]

        need_states_list_32 = [tf.cast(state, tf.float32) for state in need_states]
        x = tf.concat(need_states_list_32, axis=1)

        circuit = build_sub(str, subwire_list, range(target_qubits))
        # If needed, uncomment to visualize the subcircuit (stdout)
        # print(qml.draw(circuit)(need_draw_input, draw_weights))

        qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=target_qubits)

        x = qlayer(x)
        substates = tf.split(x, num_or_size_splits=len(subwire_list), axis=1)
        substates_list = list(substates)

        for wire in subwire_list:
            states_list[wire] = substates_list[substate_ctr]
            substate_ctr += 1

        wire_list_ctr += 1

    states_list_32 = [tf.cast(state, tf.float32) for state in states_list]
    final_measure = [states_list_32[i] for i in measure_list]
    x = tf.concat(final_measure, axis=1)

    for i in range(classical_after):
        layer = tf.keras.layers.Dense(neurons_after[i], activation=activation_after[i])
        x = layer(x)

    outputs = x
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def graph(q_history):
    plt.style.use("seaborn-v0_8")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(q_history.history["val_accuracy"], label="Cut")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(q_history.history["accuracy"], label="Cut")
    ax2.set_ylabel("Training Accuracy")
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def make_string(old_qubits):
    q_string=""
    gate_ctr=0
    single_ctr=0
    cnot_ctr=0
    hadam_list=[]
    hadamard=input("The qubits are prepared in a ground state, would you like to convert them into a superposed state using a Hadamard gate? Enter 'Y' if yes and anything else if no: ")
    if hadamard=='Y':
        hadam_list=input("Please enter all wires you wish to apply a Hadamard on. Please list them off with a comma in between each number e.g. 0,3,4. You may also use 'All' to apply it to all wires: ")
        if hadam_list=='All':
            hadam_list=range(old_qubits)
        else:
            new_hadam_list=[]
            for x in hadam_list.split(','):
                new_hadam_list.append(int(x))
            hadam_list=new_hadam_list
        for i in hadam_list:
            q_string += f"h q[{i}];\n"
    print("Here are the available gates, please use the keys provided as shown (e.g. rot, cx):")
    for key, (_, description, _) in patterns.items():
        print(f"- {key}: {description}")
    print("\nIf you wish to add a new item to the key list, please enter NEW\n")
    print("If you are finished entering all gates, please enter DONE\n")
    first_run=False
    while True:
        if first_run==False:
            key = input(f"Enter the key of gate #{gate_ctr}: ")
            first_run=True
        else:
            key = input(f"Enter the key of gate #{gate_ctr}, if complete enter DONE: ")
        if key in patterns:
            p, desc, wires  = patterns[key]
            if key=="rot":
                while True:
                    user_string = input(f"Enter the rotation gate (rx, ry, or rz) and the wire in the following format e.g. ry,0: ").strip()
                    gate, wire = user_string.split(',')
                    if gate in ['rx', 'ry','rz'] and int(wire) < old_qubits:
                        q_string+=f"{gate}(0) q[{wire}];\n"
                        single_ctr += 1
                        gate_ctr += 1
                        break
                    else:
                        print("Syntax error, try again")
                        break
            else:
                while True:
                    user_string = input(f"Enter the two wires for the CNOT gate in order e.g. 0,1 for a CNOT 0 to 1: ").strip()
                    wire_one, wire_two = user_string.split(',')
                    if int(wire_one) < old_qubits and int(wire_two)<old_qubits:
                        q_string+=f"cx q[{wire_one}],q[{wire_two}];\n"
                        cnot_ctr += 1
                        gate_ctr += 1
                        break
                    else:
                        print("Wire numbers out of range, try again")
                        break

        elif key=="DONE":
            break
        elif key=="NEW":
            add_definition_to_func_map()
        else:
            print(f"Pattern '{key}' not found in the pattern dictionary.")
            continue
    return q_string, single_ctr, cnot_ctr, hadamard, hadam_list


def add_definition_to_func_map():
    key = input("Enter the key name: ").strip()
    pattern = input("Enter the QASM string in regex pattern: ").strip()
    desc = input("Enter the description/syntax of this string: ").strip()
    num_wires = int(input("Enter the number of wires: ").strip())

    lambda_func_code = "lambda x: ("
    for i in range(1, num_wires + 1):
        if i > 1:
            lambda_func_code += ", "
        lambda_func_code += f"int(x[{i}])"
    lambda_func_code += ")"

    lambda_func = eval(lambda_func_code)

    while True:
        code_key = input(
            "Enter the function of this gate from pennylane, omit the qml. (e.g. enter 'CNOT' and not 'qml.CNOT'")
        operation = getattr(qml, code_key, None)
        if operation is None:
            print(f"Operation '{key.upper()}' not found in qml.")
            continue
        else:
            break

    func_map[key] = operation
    patterns[key] = (pattern, desc, lambda_func)
    print(f"Added {key}: {func_map[key]}")

def paste_string():
    print("Please paste your string here, enter a blank line when complete:")
    q_string=""
    while True:
        line = input()
        if line:
            q_string+=line+'\n'
        else:
            break
    sorted_circuit, ignored_string = string_sort(q_string, 1)
    print("The following lines were ignored for not being defined:")
    for i in ignored_string.split('\n'):
        print(i)
    while True:
        check=input("Would you like to make a new definition? Enter Y or N: ")
        if check == 'Y':
            repeat=int(input("Enter how many new entries you would like to make: "))
            for i in range(repeat):
                add_definition_to_func_map()
            rep=input("Do you wish to repaste your string? Enter 'Y' if yes, and anything else if no: ")
            if rep == 'Y':
                paste_string()
            break
        elif check == 'N':
            break
        else:
            print('Invalid response, please try again')
    return q_string

"""Functions for the code's encoding.

Defines functions related to computing the encodings for the code.
"""

import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, QuantumRegister

def get_random_encoding(n, k, num_gates, num_local_qubits=2, qr=None, seed=None,
                        stabilizer_format='matrix', gate_error_list=None,
                        return_logical_operators=False, return_circuit=False,
                        return_encoding=False, return_gate_errors=False):
    """
    Generates a random encoding for a stabilizer code.

    Parameters
    ----------
    n : int
        Total number of encoding qubits.
    k : int
        Number of original data qubits.
    num_gates : int
        Number of Clifford basis gates composing the encoding.
    num_local_qubits : int, optional
        Number of qubits the Clifford basis gates affect. Defaults to 2.
    qr : QuantumRegister, optional
        Optional quantum register, used to create the circuit.
    seed : int, optional
        Seed for random number generator.
    stabilizer_format : {'matrix', 'str', 'both'}, optional
        Format to return the stabilizers. 
        If 'matrix', returns them as a ((n-k) x 2n) parity check matrix.
        If 'str', they are returned as a list of strings, each indicating a stabilizer.
        If 'both', returns both.
    return_logical_operators : bool, optional
        If True, also return the 2k minimal logical operators associated with the encoding.
    return_circuit : bool, optional
        If True, also return the circuit implementing the encoding.
    return_encoding : bool, optional
        If True, also return the encoding as an instruction.
    return_gate_errors : bool, optional
        If True, also return the gate errors.

    Returns
    -------
    stabilizers : 2D array, list, tuple
        Either a ((n-k) x 2n) 2D array, or a list of strings, or a tuple of 
        both, encoding the stabilizers, depending on `stabilizer_format`.
    logical_operators : tuple
        Tuple ``(logical_Xs, logical_Zs)`` containing a list of strings 
        indicating the k Pauli strings encoding the X logical gates, and a 
        list of k Pauli strings encoding the Z logical gates.
    circuit : QuantumCircuit
        Quantum circuit implementing the encoding.
    encoding : Instruction
        Encoding given as an instruction.

    Notes
    -----
    It is assumed that the stabilizer order that Qiskit uses is preserved, so 
    that, among the n minimal stabilizers of the encoding, the first k are 
    actually logical operators (encoding the Z gate) and the next (n-k) 
    stabilizers are the ones actually considered stabilizers in the context of
    stabilizer codes. 
    """

    # Setting the seed for reproducibility
    np.random.seed(seed)

    # Creating sets of qubits to which gates will be applied
    qubit_sets_to_apply_to = [
        list(
            np.random.choice(n, num_local_qubits, replace=False)
        ) for _ in range(num_gates)
    ]

    # Creating a list of random Clifford gates
    gate_list = [
        qi.random_clifford(
            num_local_qubits, seed=seed).to_circuit() for _ in range(num_gates)
    ]

    # Initializing quantum register if none is provided
    if qr is None:
        qr = QuantumRegister(n, 'q')

    # Building the quantum circuit
    circ = QuantumCircuit(qr)
    for gate, mapping in zip(gate_list, qubit_sets_to_apply_to):
        circ.append(gate, [qr[i] for i in mapping])

    # Getting the stabilizers from the circuit
    total_stabilizers = qi.Clifford(circ).stabilizer
    stabilizers = total_stabilizers[k:]  # Here we remove the first k stabilizers

    # Processing the stabilizers according to the chosen format
    if stabilizer_format == 'str':
        stabilizers = stabilizers.to_labels()
    elif stabilizer_format == 'matrix':
        # Stabilizers in parity check matrix format
        stabilizers = stabilizers.array.astype(int)
    elif stabilizer_format == 'both':
        stabilizers_str = stabilizers.to_labels()
        parity_check_matrix = stabilizers.array.astype(int)
        stabilizers = (parity_check_matrix, stabilizers_str)

    # Initializing the list of return values with stabilizers
    return_values = [stabilizers]

    if return_logical_operators:
        h_circ = QuantumCircuit(qr)
        h_circ.h([qr[i] for i in range(k)])
        result_X = qi.Clifford(h_circ + circ).stabilizer.to_labels()

        result_Z = total_stabilizers.to_labels()
        logical_Zs = result_Z[:k]
        logical_Xs = result_X[:k]
        return_values.append((logical_Xs, logical_Zs))

    if return_circuit or return_encoding:
        encoding = circ.to_instruction()
        if return_circuit:
            circuit = QuantumCircuit(qr)
            circuit.append(encoding, qr)
            return_values.append(circuit)

        if return_encoding:
            return_values.append(encoding)

    if return_gate_errors:
        gate_errors = get_gate_errors(n, gate_list, qubit_sets_to_apply_to, gate_error_list=gate_error_list)
        return_values.append(gate_errors)

    return tuple(return_values)

def get_gate_errors(n, gate_list, qubit_sets_to_apply_to, gate_error_list=None):
    """
    Given a number of qubits and a list of gates, compute the gate errors 
    for specific qubits sets. 

    Parameters
    ----------
    n : int
        The number of qubits.
    gate_list : list
        The list of gates.
    qubit_sets_to_apply_to : list
        The list of sets of qubits to which the gates are to be applied.
    gate_error_list : list, optional
        The list of gate errors. If not provided, it defaults to a specific list.

    Returns
    -------
    list
        List of evolved gate errors after each gate operation.
    """
    # Default gate errors
    if gate_error_list is None:
        gate_error_list = ['IX','IY','IZ',
                            'XI','XX','XY','XZ',
                            'YI','YX','YY','YZ',
                            'ZI','ZX','ZY','ZZ']

    # Converting gate list to Clifford objects
    cliff_gate_list = [qi.Clifford(gate) for gate in gate_list]

    # Initializing quantum circuit
    qr = QuantumRegister(n, 'q')
    circ = QuantumCircuit(qr)

    # Initializing Clifford objects for the circuit
    mid_gate = qi.Clifford(circ)

    new_error_list = []
    for i, (gate, mapping) in enumerate(zip(reversed(cliff_gate_list), reversed(qubit_sets_to_apply_to))):
        new_error_list.append([])
        for error in gate_error_list:
            # Convert the error to qiskit error
            exp_error = _error_to_qiskit(error, mapping, n)
            # Evolve the mid_gate with the error
            new_error = exp_error.evolve(mid_gate, frame='s')
            new_error_list[i].append(new_error.to_label())
        # Compose the mid_gate with the current gate for the next iteration
        mid_gate = mid_gate.compose(gate, qargs=mapping, front=True)

    return new_error_list[::-1]

def _error_to_qiskit(error, mapping, n):
    """
    Convert a given error into a Qiskit error, according to a mapping.

    Parameters
    ----------
    error : str
        The error string.
    mapping : list
        The mapping list for qubits.
    n : int
        The number of qubits.

    Returns
    -------
    Pauli
        The corresponding Qiskit Pauli object for the error.
    """
    # Initialize an identity Pauli operation for all qubits
    pauli = ['I'] * n
    list_error = list(error)
    for op, qb in zip(list_error, mapping):
        pauli[n - qb - 1] = op

    # Create the new error as a Pauli operation
    new_error = ''.join(pauli)
    return qi.Pauli(new_error)

def get_random_parity_check_matrix(n, s):
    """
    Creates a random 2n x s parity check matrix. 

    Parameters
    ----------
    n : int
        Number of qubits.
    s : int
        Number of stabilizers.

    Returns
    -------
    2D array
        2n x s parity check matrix.
    """
    # Number of logical qubits
    k = n - s
        
    # Generate a random Clifford and get the stabilizer generators
    # Convert to int array and return
    parity_check_matrix = qi.random_clifford(int(n)).stabilizer[k:].array.astype(int)

    return parity_check_matrix

def get_statistics(n, k, num_gates, num_local_qubits=2):
    """
    Given a number of qubits, logical qubits and gates, compute the statistics 
    of how many line stats are influenced by each gate.

    Parameters
    ----------
    n : int
        The number of qubits.
    k : int
        The number of logical qubits.
    num_gates : int
        The number of gates.
    num_local_qubits : int, optional
        The number of local qubits to be randomly chosen for each gate. Defaults to 2.

    Returns
    -------
    ndarray
        Array of length (n-k+1), each index i contains the count of how many line stats have size i.
    """
    # Initialize line stats
    line_stats = {}
    for i in range(n):
        if i < k:
            line_stats[i] = set()
        else:
            line_stats[i] = set([i])

    # Initialize count stats
    count_stats = np.zeros(n-k+1, dtype=int)

    # Iterate over the gates
    for _ in range(num_gates):
        # Randomly select local qubits for each gate
        mapping = np.random.choice(n, num_local_qubits, replace=False)
        # Get the union of the current line stats for these qubits
        s = set().union(*[line_stats[i] for i in mapping])
        # Update the line stats for these qubits
        for i in mapping:
            line_stats[i] = s
        # Update the count stats for the size of the current line stats
        count_stats[len(s)] += 1

    return count_stats

from .DegenerateFaultTolerance import DegenerateFaultTolerance
from .ldpc import get_random_QLDPC_parity_check_matrix, pauli_H
from qiskit import QuantumRegister, QuantumCircuit

class LDPCFaultTolerance(DegenerateFaultTolerance):
    """
    Class representing a Low-Density Parity-Check (LDPC) fault tolerance scheme
    based on the `DegenerateFaultTolerance` class.
    
    Parameters
    ----------
    n : int
        Size of the quantum register.
    k : int
        Number of ancillary qubits.
    c : int
        Weight of the random parity check matrix.
    gate_error_list : list, optional
        List of gate errors to be applied.
    apply_stabilizer_errors : bool, optional
        Whether to apply stabilizer errors.
    """

    def __init__(self, n, k, c, gate_error_list=None, apply_stabilizer_errors=True):
        # Initialize attributes
        self.n, self.k, self.c = n, k, c
        self.qr = QuantumRegister(n, 'q')  # Quantum register
        self.anc = QuantumRegister(n-k, 'a')  # Ancillary register
        self._set_gate_error_list(gate_error_list)  # Set gate errors
        self.full_circuit = None  # Full quantum circuit
        # Applying encoding errors doesn't make sense in this context, so it's always False
        self.apply_encoding_errors = False
        self.apply_stabilizer_errors = apply_stabilizer_errors
        self._set_bin_rep()

    def get_encoding(self, timeout=300):
        """
        Generate encoding circuit using a random parity-check matrix and Pauli-H transformation.
        
        Parameters
        ----------
        timeout : int, optional
            Timeout for the generation of the random parity-check matrix.
        """
        # Generate random QLDPC parity check matrix
        random_matrix = get_random_QLDPC_parity_check_matrix(self.n, self.k, self.c, timeout=timeout)
        # Apply Pauli-H transformation to the random matrix to get stabilizers
        self.stabilizers = pauli_H(random_matrix)
        # Define encoding circuit
        self.encoding = QuantumCircuit(self.qr)


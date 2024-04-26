from .encoding import get_random_encoding, _error_to_qiskit
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.quantum_info as qi
import re
import numpy as np
from scipy.special import comb
from itertools import combinations, product
import warnings

class FaultTolerance:
    """
    A class to model and simulate fault-tolerant quantum computations.

    Attributes:
    -----------
    n: int
        The number of qubits.
    k: int
        The number of qubits after encoding.
    num_gates: int
        The number of gates.
    num_local_qubits: int
        The number of local qubits. Defaults to 2.
    gate_error_list: list
        The list of gate errors.
    seed: int
        The seed for the random number generator.
    apply_encoding_errors: bool
        A flag to apply encoding errors or not. Defaults to False.
    apply_stabilizer_errors: bool
        A flag to apply stabilizer errors or not. Defaults to True.
    use_row_echelon_form: bool
        A flag to use row echelon form or not. Defaults to False.
    """

    def __init__(self,
                n,
                k,
                num_gates,
                num_local_qubits = 2,
                gate_error_list = None,
                seed = None,
                apply_encoding_errors = False,
                apply_stabilizer_errors = True,
                use_row_echelon_form = False):
        """
        Constructs all the necessary attributes for the FaultTolerance object.

        Parameters:
        -----------
        n, k, num_gates: int
            The parameters for the fault-tolerant system.
        num_local_qubits: int, optional
            The number of local qubits. Defaults to 2.
        gate_error_list: list, optional
            The list of gate errors. Defaults to None.
        seed: int, optional
            The seed for the random number generator. Defaults to None.
        apply_encoding_errors: bool, optional
            A flag to apply encoding errors or not. Defaults to False.
        apply_stabilizer_errors: bool, optional
            A flag to apply stabilizer errors or not. Defaults to True.
        use_row_echelon_form: bool, optional
            A flag to use row echelon form or not. Defaults to False.
        """
        self.n = n
        self.k = k
        self.num_gates = num_gates
        self.num_local_qubits = num_local_qubits
        self.qr = QuantumRegister(n, 'q')
        self.anc = QuantumRegister(n - k, 'a')
        self.seed = seed
        self._set_gate_error_list(gate_error_list)
        self.full_circuit = None
        self.apply_encoding_errors = apply_encoding_errors
        self.apply_stabilizer_errors = apply_stabilizer_errors
        self.use_row_echelon_form = use_row_echelon_form
        self._set_bin_rep()

    def _set_bin_rep(self):
        """
        Method to set binary representation of the gate error list.
        """
        size = len(self.gate_error_list)
        bin_rep = np.zeros((2 ** size, size), dtype=int)
        for i in range(bin_rep.shape[0]):
            bin_rep[i] = np.array(list(np.binary_repr(i).zfill(size))).astype(int)
        self.bin_rep = bin_rep

    def _set_gate_error_list(self, gate_error_list):
        """
        Method to set gate error list.

        Parameters:
        -----------
        gate_error_list: list
            The list of gate errors. If none is given, a default list is assigned.
        """
        # Qubit order is increasing
        # Since control is generally the lower index, then the first case is the
        # control
        if gate_error_list is None:
            # Default gate error list
            self.gate_error_list = ['IX', 'IZ', 'XI', 'ZI']
        else:
            self.gate_error_list = gate_error_list

    def get_encoding(self):
        """
        Method to get the encoding of the quantum system.

        Uses random encoding and sets the stabilizers and encoding for the object.
        """
        self.stabilizers, self.encoding = get_random_encoding(self.n,
                                                              self.k,
                                                              self.num_gates,
                                                              self.num_local_qubits,
                                                              self.qr,
                                                              seed=self.seed,
                                                              stabilizer_format='str',
                                                              return_circuit=True)
        
        # Decomposing the circuit twice to break it down to the simplest gates
        self.encoding = self.encoding.decompose().decompose()

    def get_syndrome_circuit(self, apply_barriers=False):
        """
        Generates a quantum circuit for calculating syndromes.

        Parameters
        ----------
        self : object
            Self-reference object.
        apply_barriers : bool, optional
            Whether or not to apply barriers between each operation. Default is False.

        Returns
        -------
        None

        Notes
        -----
        Updates the `syndrome_circuit` attribute of the instance.
        """
        # Initialize the ancilla and quantum registers
        anc, qr = self.anc, self.qr

        # Create a quantum circuit using the ancilla and quantum registers
        syndrome_circuit = QuantumCircuit(anc, qr)

        # Apply a Hadamard gate to the ancilla register
        syndrome_circuit.h(anc)

        # Apply a barrier if specified
        if apply_barriers:
            syndrome_circuit.barrier(anc)

        # Loop through the stabilizers
        for i, stab in enumerate(self.stabilizers):
            # Apply a barrier before starting a new stabilizer if specified and it's not the first one
            if i != 0 and apply_barriers:
                syndrome_circuit.barrier(anc, qr)

            # Loop through the reversed stabilizer
            for j, pauli in enumerate(reversed(list(stab)[1:])):
                # Depending on the type of Pauli operator, apply a specific gate or sequence of gates
                if pauli == 'X':
                    syndrome_circuit.cx(anc[i], qr[j])
                elif pauli == 'Y':
                    syndrome_circuit.sdg(qr[j])
                    syndrome_circuit.cx(anc[i], qr[j])
                    syndrome_circuit.s(qr[j])
                elif pauli == 'Z':
                    syndrome_circuit.h(qr[j])
                    syndrome_circuit.cx(anc[i], qr[j])
                    syndrome_circuit.h(qr[j])

        # Apply a barrier and a Hadamard gate to the ancilla register if specified
        if apply_barriers:
            syndrome_circuit.barrier(anc)

        syndrome_circuit.h(anc)

        # If the first character of the stabilizer is '-', apply an X gate to the ancilla
        for i, stab in enumerate(self.stabilizers):
            if stab[0] == '-':
                syndrome_circuit.x(anc[i])

        # Apply a final barrier to the ancilla register if specified
        if apply_barriers:
            syndrome_circuit.barrier(anc)

        # Convert the quantum circuit to an instruction and store it in the instance variable
        self.syndrome_circuit = syndrome_circuit.to_instruction()

    def get_full_circuit(self, n_stab_runs=1):
        """
        Create a full quantum circuit for error correction. 

        Parameters
        ----------
        n_stab_runs : int, optional
            The number of stabilizer runs, by default 1.

        """
        # Construct ancillary registers
        ancs = list(reversed([QuantumRegister(self.n-self.k, 'a'+str(i)) for i in range(n_stab_runs)]))
        
        # Construct full quantum circuit with ancillary registers and quantum register
        full_circuit = QuantumCircuit(*(ancs + [self.qr]))
        
        # Append the encoding to the quantum register
        full_circuit.append(self.encoding, self.qr)

        # Add syndrome circuit to each ancillary register
        for reg in reversed(ancs):
            full_circuit.append(self.syndrome_circuit, (*reg, *self.qr))

        # Decompose the circuit to elementary operations
        full_circuit = full_circuit.decompose()

        # Create a full quantum register
        self.full_qr = QuantumRegister(self.n + n_stab_runs*(self.n-self.k), 'f')
        
        # Construct the final full circuit
        self.full_circuit = QuantumCircuit(self.full_qr)
        self.full_circuit.compose(full_circuit, inplace=True)

        # Update total number of qubits
        self.n_total = self.full_circuit.num_qubits

    def evolve_gate_errors(self, ignore_extra_stabs=0):
        """
        Evolve the gate errors in the circuit.

        Parameters
        ----------
        ignore_extra_stabs : int, optional
            The number of extra stabilizers to ignore, by default 0.
        """
        self._ignore_extra_stabs = ignore_extra_stabs

        # Convert the full circuit to QASM string
        full_circuit_qasm = self.full_circuit.qasm()

        # Split the QASM string into lines
        split_qasm = full_circuit_qasm.split('\n')

        # Extract the initial QASM lines
        initial_qasm = split_qasm[:3]

        # Initialize list for evolved errors
        ev_error_list = []

        # Iterate over each QASM line (reversed) and identify the CNOT gates
        mid_qasm = []
        for line in reversed(split_qasm[3:]):
            if 'cx' in line:
                loc = re.findall('f\[(\d+)\]', line)
                mapping = [int(d) for d in loc]
                add_cx = self._should_add_cx(mapping, ignore_extra_stabs)

                if add_cx:
                    full_qasm = initial_qasm + list(reversed(mid_qasm))
                    mid_circuit = QuantumCircuit.from_qasm_str('\n'.join(full_qasm))
                    mid_gate = qi.Clifford(mid_circuit)
                    ev_errors = self._get_evolved_errors(mapping, mid_gate)
                    ev_error_list.append(ev_errors)

            mid_qasm.append(line)

        self.ev_error_list = ev_error_list
        self.n_cx = len(ev_error_list)

    def _should_add_cx(self, mapping, ignore_extra_stabs):
        """
        Determine if a CNOT gate should be added based on its qubit mapping.

        Parameters
        ----------
        mapping : list
            The qubit mapping for the CNOT gate.
        ignore_extra_stabs : int
            The number of extra stabilizers to ignore.

        Returns
        -------
        bool
            Whether to add the CNOT gate.
        """
        add_cx = True
        if not self.apply_encoding_errors and min(mapping) >= self.n_total - self.n:
            add_cx = False
        if not self.apply_stabilizer_errors and min(mapping) < self.n_total - self.n:
            add_cx = False
        else:
            if min(mapping) < ignore_extra_stabs*(self.n-self.k):
                add_cx = False
        return add_cx

    def _get_evolved_errors(self, mapping, mid_gate):
        """
        Get the evolved errors based on the error mapping and gate evolution.

        Parameters
        ----------
        mapping : list
            The qubit mapping for the CNOT gate.
        mid_gate : qi.Clifford
            The mid-gate after which the errors evolve.

        Returns
        -------
        list
            The evolved errors.
        """
        ev_errors = []
        for error in self.gate_error_list:
            exp_error = _error_to_qiskit(error, mapping, self.n_total)
            new_error = exp_error.evolve(mid_gate, frame='s')
            ev_errors.append(new_error.to_label())
        return ev_errors

    def _get_syndrome(self, error_str):
        """
        Get the syndrome for a given error string.

        Parameters
        ----------
        error_str : str
            The error string.

        Returns
        -------
        np.ndarray
            The syndrome for the error string.
        """
        error = list(error_str.lstrip('-'))
        syndrome = np.zeros(self.n_total - self.n, dtype=int)
        for i, op in enumerate(error[self.n:]):
            if op in ['X','Y']:
                syndrome[i] = 1
        return syndrome

    def get_base_syndromes(self, show=False):
        """
        Compute and store the base syndromes for the circuit.

        Parameters
        ----------
        show : bool, optional
            Whether to print the syndromes, by default False.
        """
        self.base_syndromes = np.zeros((self.n_cx, len(self.gate_error_list), self.n_total-self.n), dtype=int)
        for i, ev_errors in enumerate(self.ev_error_list):
            for j, error in enumerate(ev_errors):
                self.base_syndromes[i,j] = self._get_syndrome(error)
                if show:
                    print(i, j, error, self.base_syndromes[i,j])

    def syndrome_generator(self, max_t, n_errors):
        """
        Generator for syndromes based on the error combinations.

        Parameters
        ----------
        max_t : int
            The maximum number of errors.
        n_errors : int
            The number of errors.

        Yields
        -------
        tuple
            The error number and corresponding syndrome.
        """
        n_cx = self.n_cx
        array = np.zeros(self.n_total-self.n, dtype=int)
        for dd in range(1, max_t+1):
            case = product(product(range(1, n_errors+1), repeat=dd), combinations(range(n_cx), dd))
            for error_nums, inds in case:
                for ind_err, ind_cx in zip(error_nums, inds):
                    syndrome = self.syndrome_power_set(ind_err, ind_cx)
                    array = (array + syndrome) % 2
                yield dd, array
                array.fill(0)

    def syndrome_power_set(self, ind_err, ind_cx):
        """
        Get the syndrome for a power set of errors.

        Parameters
        ----------
        ind_err : int
            The index of the error.
        ind_cx : int
            The index of the CNOT gate.

        Returns
        -------
        np.ndarray
            The syndrome for the power set of errors.
        """
        # Get syndromes corresponding to the CNOT gate index
        syndromes = self.base_syndromes[ind_cx]
        array = np.zeros(self.n_total - self.n, dtype=int)
        # Iterate over binary representation of error index
        for i, ind in enumerate(self.bin_rep[ind_err]):
            if ind == 1:
                array = (array + syndromes[i]) % 2
        return array

    def get_weight_statistics(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=None,
                            truncate=True):
        """
        Calculate statistics for the weight of quantum error correction codes.

        This function acts as a lower bound for the performance of the code, 
        disregarding potential degeneracies in the main n qubits and ancilla qubits.

        Parameters
        ----------
        prob : float or list, optional
            Error probabilities, by default None.
        max_t : int, optional
            Maximum error weight, by default None.
        fill_syndromes : bool, optional
            Whether to fill the syndrome list, by default False.
        n_cx_errors : int, optional
            Number of CX errors, by default None.
        truncate : bool, optional
            If True, number of visible errors is limited to max_t. 
            Otherwise, it is equal to n_cx, by default True.

        Returns
        -------
        tuple
            Failure rates, correctable fraction, and mean iterations of the error correction code.
        """
        
        # Warn users about function's assumptions
        warnings.warn("""This function is a lower bound for the performance 
                        of the code. It assumes that errors are distinct. 
                        Therefore, it not only disregards possible degeneracies
                        in the main n qubits, but also the highly degenerate 
                        errors in the ancilla qubits.""")

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        # Set default values if None
        n_cx_errors = len(self.gate_error_list)**2 - 1 if n_cx_errors is None else n_cx_errors
        max_t = n_cx if max_t is None else max_t
        n_show = max_t if truncate else n_cx

        # Ensure prob is a numpy array
        if isinstance(prob, float):
            prob = np.array([prob])
        elif not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        # Initialize syndrome list or set
        if fill_syndromes:
            syndrome_list = np.zeros(2**(n_total-n), dtype=bool)
            syndrome_list[0] = True
        else:
            seen_syndromes = set([0])

        accounted_syndromes = 1
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])

        # Process syndromes
        for dd, syndrome in self.syndrome_generator(max_t, n_cx_errors):
            #int_syndrome = sum(bool(bit) << i for i, bit in enumerate(syndrome[::-1]))
            out = 0
            for bit in syndrome:
                out = (out << 1) | bool(bit)
            int_syndrome = out

            if fill_syndromes and not syndrome_list[int_syndrome] or int_syndrome not in seen_syndromes:
                if fill_syndromes:
                    syndrome_list[int_syndrome] = True
                else:
                    seen_syndromes.add(int_syndrome)

                accounted_syndromes += 1
                error_type[dd-1] += 1
                probabilities[dd-1] += (prob/n_cx_errors)**dd * (1-prob)**(n_cx-dd)

                # Break loop if all syndromes are accounted for
                if accounted_syndromes == 2**(n_total-n):
                    break

        # Compute failure rate
        failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

        # Compute the number of errors
        n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
        correctable_fraction = error_type / n_errors

        # Compute the mean number of iterations
        mean_iterations = np.array([
            sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                    sum(error_type[:dd]) + (nn+1)/2
                ) for dd, nn in enumerate(error_type[:i])
            ) for i in range(1, n_show+1)
        ])

        return failure_rate, correctable_fraction, mean_iterations
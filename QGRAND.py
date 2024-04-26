import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer#, transpile
#from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
import qiskit as qk

#from IPython.display import clear_output

#import matplotlib.pyplot as plt
#plt.rc('text',usetex=True)
#plt.rc('font', family='serif', size=16)
#plt.rcParams['figure.figsize'] = [10, 6]
#from datetime import datetime
#from time import time

from scipy.interpolate import interp1d
from scipy.special import comb
from itertools import combinations, product

from .hamming import hamming_bound, max_distance_hamming
from .encoding import get_random_encoding
from .pauli import pauli_generator, get_numbered_noise

class QGRAND:

    def __init__(self,
                n=None,
                k=None,
                num_gates=None,
                num_local_qubits=2,
                noise_statistics=None,
                max_iterations=None,
                compress_circuit=False,
                backend=None):

        self.n, self.k, self.num_gates = n, k, num_gates
        self.num_local_qubits = num_local_qubits

        if noise_statistics is not None:
            if isinstance(noise_statistics[0], str):
                self.noise = noise_statistics
                self.noise_probabilities = [1/len(self.noise) for _ in self.noise]
                self.noise_statistics = [(prob, error) for prob, error in zip(self.noise_probabilities, self.noise)]

            else:
                self.noise_statistics = noise_statistics
                self.noise = [error for prob, error in noise_statistics]
                self.noise_probabilities = [prob for prob, error in noise_statistics]

        self.qb = QuantumRegister(n, 'q')
        self.anc = QuantumRegister(n-k, 'a')
        self.cb = ClassicalRegister(n-k, 'c')
        self.max_iterations = max_iterations if max_iterations is not None else 3*n+2
        self.compress_circuit = compress_circuit

        if backend is None:
            self.backend = Aer.get_backend('aer_simulator_stabilizer')
        else:
            self.backend = Aer.get_backend(backend)

        self.syndrome_circuit = None
        self.initial_circuit = None

        self._hamming_f = self._get_hamming_bound_interp()

    @staticmethod
    def _get_hamming_bound_interp():
        code_array = np.linspace(0,1,101)
        h_list = hamming_bound(code_array)
        _hamming_f = interp1d(h_list, code_array)
        return _hamming_f


    def get_encoding(self, fast=False):

        if fast:
            stabs, self.encoding = get_random_encoding(self.n,
                                            self.k,
                                            self.num_gates,
                                            qr = self.qb,
                                            stabilizer_format = 'both',
                                            return_logical_operators = False,
                                            return_circuit = False,
                                            return_encoding = False)
            self.parity_check_matrix, self.stabilizers = stabs

        else:
            stabs, los, self.circuit, self.encoding = get_random_encoding(
                                            self.n,
                                            self.k,
                                            self.num_gates,
                                            qr = self.qb,
                                            stabilizer_format = 'both',
                                            return_logical_operators = True,
                                            return_circuit = True,
                                            return_encoding = True)
            self.parity_check_matrix, self.stabilizers = stabs
            self.logical_Xs, self.logical_Zs = los

    def get_distance_from_stabilizers(self, rerun=False, by_matrix=False):
        n, k = self.n, self.k

        try:
            hamm = self.hamming_distance
        except AttributeError:
            self.hamming_distance = max_distance_hamming(n,k, self._hamming_f)
            hamm = self.hamming_distance

        distance_Z = min([n - op.count('I') for op in self.logical_Zs])
        distance_X = min([n - op.count('I') for op in self.logical_Xs])
        
        distance_max = min(distance_Z, distance_X, hamm)
        # print("Hamming bound is d = {}.".format(hamm))
        if distance_max == 1:
            return distance_max
        else:
            if by_matrix:
                if not rerun:
                    try:
                        parity_check_matrix = self.parity_check_matrix
                    except AttributeError:
                        self.get_parity_check_matrix()
                        parity_check_matrix = self.parity_check_matrix
                else:
                    self.get_parity_check_matrix()
                    parity_check_matrix = self.parity_check_matrix

                for dd, error in pauli_generator(n, distance_max):
                    error_vector = np.zeros((1, 2*n), dtype=int)
                    for j, op in enumerate(list(error)):
                        error_vector[0, j] = 1 if op == 'Z' or op == 'Y' else 0
                        error_vector[0, n+j] = 1 if op == 'X' or op == 'Y' else 0
                    if np.sum(error_vector @ parity_check_matrix.T % 2) == 0:
                        return dd
                return distance_max

            else:
                stabilizers = qi.StabilizerTable.from_labels(self.stabilizers)
                for dd, error in pauli_generator(n, distance_max):
                    if len(stabilizers.anticommutes_with_all(error)) == 0:
                        distance = dd
                        return distance
                return distance_max

    def get_distance(self, rerun=False, by_matrix=False):
        if rerun:
            self.get_encoding()
        return self.get_distance_from_stabilizers(rerun=rerun, by_matrix=by_matrix)

    def apply_error(self, error=None):
        if error is None:
            errors = [''.join([np.random.choice(['X','Y','Z']), str(np.random.randint(self.qb.size))])]
        else:
            errors = error.split()
        self.errors = errors
        circuit = QuantumCircuit(self.qb)
        for err in errors:
            if err[0] == 'X':
                circuit.x(self.qb[int(err[1:])])
            elif err[0] == 'Z':
                circuit.z(self.qb[int(err[1:])])
            elif err[0] == 'Y':
                circuit.y(self.qb[int(err[1:])])
        self.error_circuit = circuit.to_instruction()
        self.circuit.barrier(self.qb)
        self.circuit.append(self.error_circuit, self.qb)

    def apply_stabilizers(self):
        anc, qb = self.anc, self.qb
        
        self.circuit.barrier(anc, qb)

        if self.syndrome_circuit is None:
            syndrome_circuit = QuantumCircuit(anc,qb)
            syndrome_circuit.h(anc)
            syndrome_circuit.barrier(anc)
            
            for i, stab in enumerate(self.stabilizers):
                syndrome_circuit.barrier(anc, qb) if i!=0 else None
                for j, pauli in enumerate(reversed(list(stab)[1:])):
                    if pauli == 'X':
                        syndrome_circuit.cx(anc[i], qb[j])
                    elif pauli == 'Y':
                        syndrome_circuit.cy(anc[i], qb[j])
                    elif pauli == 'Z':
                        syndrome_circuit.cz(anc[i], qb[j])
            syndrome_circuit.barrier(anc)
            syndrome_circuit.h(anc)
            for i, stab in enumerate(self.stabilizers):
                if stab[0] == '-':
                    syndrome_circuit.x(anc[i])
            syndrome_circuit.barrier(anc)

            self.syndrome_circuit = syndrome_circuit.to_instruction()

        self.circuit.append(self.syndrome_circuit, (*anc,*qb))


    def apply_error_correction(self, previous_error, next_error):
        qb, cb = self.qb, self.cb
        n, k = self.n, self.k
        correction_circuit = QuantumCircuit(qb, cb)

        if previous_error is not None:
            if previous_error[0] == 'X':
                correction_circuit.x(qb[int(previous_error[1:])])
            elif previous_error[0] == 'Z':
                correction_circuit.z(qb[int(previous_error[1:])])
            elif previous_error[0] == 'Y':
                correction_circuit.y(qb[int(previous_error[1:])])

        if next_error is not None:
            for res in range(1, 2**(n-k)):
                if next_error[0] == 'X':
                    correction_circuit.x(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Z':
                    correction_circuit.z(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Y':
                    correction_circuit.y(qb[int(next_error[1:])]).c_if(cb, res)

        self.correction_circuit = correction_circuit.to_instruction()

        self.circuit.append(self.correction_circuit, qb, cb)

    def undo_correction_attempt(self, previous_error):
        qb, cb = self.qb, self.cb
        n, k = self.n, self.k
        correction_circuit = QuantumCircuit(qb, cb)

        if previous_error[0] == 'X':
            correction_circuit.x(qb[int(previous_error[1:])])
        elif previous_error[0] == 'Z':
            correction_circuit.z(qb[int(previous_error[1:])])
        elif previous_error[0] == 'Y':
            correction_circuit.y(qb[int(previous_error[1:])])

        self.correction_circuit = correction_circuit.to_instruction()

        self.circuit.append(self.correction_circuit, qb, cb)

    def try_correction(self, next_error, conditional=True):
        qb, cb = self.qb, self.cb
        n, k = self.n, self.k
        correction_circuit = QuantumCircuit(qb, cb)

        if conditional:

            for res in range(1, 2**(n-k)):
                if next_error[0] == 'X':
                    correction_circuit.x(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Z':
                    correction_circuit.z(qb[int(next_error[1:])]).c_if(cb, res)
                elif next_error[0] == 'Y':
                    correction_circuit.y(qb[int(next_error[1:])]).c_if(cb, res)

        else:
            if next_error[0] == 'X':
                correction_circuit.x(qb[int(next_error[1:])])
            elif next_error[0] == 'Z':
                correction_circuit.z(qb[int(next_error[1:])])
            elif next_error[0] == 'Y':
                correction_circuit.y(qb[int(next_error[1:])])

        self.correction_circuit = correction_circuit.to_instruction()

        self.circuit.append(self.correction_circuit, qb, cb)

    def set_circuit(self, force=False):

        if self.initial_circuit is None:
            circuit = QuantumCircuit(self.qb, self.anc, self.cb)
            circuit.append(self.encoding, self.qb)
            circuit.append(self.error_circuit, self.qb)
            self.initial_circuit = circuit

        if force:
            self.circuit = self.initial_circuit.copy()


    def apply_QGRAND(self):
        #n, k = self.n, self.k
        self.results = []
        anc, cb, qb = self.anc, self.cb, self.qb
        n, k = self.n, self.k
        noise = ''.join(['I' for _ in range(n)]) + self.noise
        self.set_circuit()
        success = False
        ind = 0
        
        for ii in range(self.max_iterations):

            self.apply_stabilizers()

            self.circuit.measure(anc, cb)
            
            job = qk.execute(self.circuit, shots=1, backend=self.backend)
            last_result = list(job.result().get_counts().keys())[0]
            numbered_noise = get_numbered_noise(noise[ind])
            print("Iteration: {}\t Testing: {}\t Syndrome: {}".format(ii, numbered_noise, last_result))
            self.results.append((ii, last_result))
            
            if last_result == '0'*(n-k):
                print('QGRAND has corrected the error after {} iterations.'.format(ii+1))
                print('The corrected error was '+numbered_noise+'.')
                success = True
                break
                
            self.circuit.reset(anc)

            #self.apply_error_correction(noise[ind] if ind>0 else None, noise[ind+1] if ind<len(noise) else None)
            self.undo_correction_attempt(noise[ind]) if ind>0 else None
            self.try_correction(noise[ind+1]) if ind<len(noise)-1 else None
            ind += 1
            if ind == len(noise):
                break
         
        if not success:        
            print('QGRAND could not correct the error in {} iterations.'.format(self.max_iterations))

    def apply_QGRAND_fast(self):
        #n, k = self.n, self.k
        self.results = []
        anc, cb, qb = self.anc, self.cb, self.qb
        n, k = self.n, self.k
        noise = ['I'] + self.noise
        success = False
        ind = 0

        # self.apply_stabilizers()
        # clifford = qi.Clifford(self.circuit)
        self.set_circuit()
        self.circuit = self.initial_circuit.copy()

        for ii in range(self.max_iterations-1):

            self.apply_stabilizers()

            self.circuit.measure(anc, cb)
            
            job = qk.execute(self.circuit, shots=1, backend=self.backend)
            last_result = list(job.result().get_counts().keys())[0]
            print("Iteration: {}\t Testing: {}\t Syndrome: {}".format(ii, noise[ind], last_result))
            self.results.append((ii, last_result))
            
            if last_result == '0'*(n-k):
                print('QGRAND has corrected the error after {} iterations.'.format(ii+1))
                print('The corrected error was '+noise[ind]+'.')
                success = True
                break

            self.circuit = self.initial_circuit.copy()
                
            self.try_correction(noise[ind+1], conditional=False) if ind<len(noise)-1 else None
            ind += 1
            if ind == len(noise):
                break
         
        if not success:        
            print('QGRAND could not correct the error in {} iterations.'.format(self.max_iterations))

    def get_parity_check_matrix(self):
        """
        This method generates the parity check matrix.

        The parity check matrix is generated based on the stabilizers of the class instance.
        The method populates a 2D numpy array with integers.

        """

        n = self.n
        parity_check_matrix = np.zeros((len(self.stabilizers), 2*n), dtype=int)

        # Iterating over stabilizers
        for i, stabilizer in enumerate(self.stabilizers):
            # We start from index 1 as per original code because the first stabilizer is the sign, 
            # which doesn't influence the parity check matrix
            for j, pauli in enumerate(list(stabilizer)[1:]):
                if pauli == 'I':
                    continue
                else:
                    if pauli != 'Z':
                        parity_check_matrix[i, j] = 1
                    if pauli != 'X':
                        parity_check_matrix[i, n+j] = 1

        self.parity_check_matrix = parity_check_matrix

    def get_syndrome_table(self, error_matrix=None):
        """
        This method generates the syndrome table.

        The syndrome table is generated by multiplying the error matrix by the transpose of the parity check matrix.

        Parameters
        ----------
        error_matrix : np.array, optional
            The error matrix, if not provided, it's obtained from the instance.
        """

        if error_matrix is None:
            error_matrix = getattr(self, "error_matrix", None)
            if error_matrix is None:
                self.get_error_matrix()
                error_matrix = self.error_matrix

        try:
            self.syndrome_table = error_matrix @ self.parity_check_matrix.T % 2
        except AttributeError:
            self.get_parity_check_matrix()
            self.syndrome_table = error_matrix @ self.parity_check_matrix.T % 2

    def get_int_syndrome_vector(self):
        """
        This method converts the syndrome table to integer vector representation.

        Each syndrome is converted to an integer value.
        """

        int_syndrome_vector = np.zeros(self.syndrome_table.shape[0], dtype=int)

        # Convert each syndrome to an integer
        for i, syndrome in enumerate(self.syndrome_table):
            out = 0
            for bit in syndrome:
                out = (out << 1) | bit
            int_syndrome_vector[i] = out

        self.int_syndrome_vector = int_syndrome_vector

    def get_error_to_syndrome_mapping(self):
        """
        This method generates a mapping from error to syndrome.

        The mapping is created by associating each error to its corresponding syndrome.
        """

        self.error_to_syndrome_mapping = [(syndrome, get_numbered_noise(error)) 
                                          for syndrome, error in zip(self.int_syndrome_vector, self.noise)]

    def get_error_to_syndrome_prob_mapping(self):
        """
        This method generates a mapping from error to syndrome and their corresponding probability.

        The mapping is created by associating each error to its corresponding syndrome and probability.
        """

        self.error_to_syndrome_prob_mapping = [
        (syndrome, get_numbered_noise(error), prob) 
        for syndrome, error, prob in zip(self.int_syndrome_vector, self.noise, self.noise_probabilities)]


    def get_syndrome_table_with_leaders(self):
        """
        Generates a sorted syndrome table with leaders.

        The method sorts the error-to-syndrome mapping and groups them by syndrome. 
        Each group is then sorted by error probabilities to identify the most probable error 
        (leader) for each syndrome.

        Updates the attribute `syndrome_table_with_leaders` with the final sorted table.
        """
        
        # Sort the error-to-syndrome mapping
        sorted_table = sorted(self.error_to_syndrome_prob_mapping, key=lambda x: x[0])

        table_with_leaders = []
        syndrome = None
        for error in sorted_table:
            error_syndrome, error_str, prob = error
            if error_syndrome != syndrome:
                syndrome = error_syndrome
                # Add a new syndrome group when the syndrome changes
                table_with_leaders.append([error_syndrome, []])
            # Append the error to the current syndrome group
            table_with_leaders[-1][-1].append((error_str, prob))

        final_table = []
        for syndrome, errors in table_with_leaders:
            # Sort errors in each group by their probability (in descending order)
            sorted_errors = sorted(errors, key=lambda x: -x[1])
            prob = sorted_errors[0][1]
            only_errors = [error for error, _ in sorted_errors]
            # Append the syndrome, the highest error probability, and the sorted errors to the final table
            final_table.append([syndrome, prob, only_errors])

        # Update the attribute with the final sorted table
        self.syndrome_table_with_leaders = sorted(final_table, key=lambda x: -x[1])


    def get_error_rate(self, p=None):
        """
        Calculates the failure rate based on the syndrome table with leaders.

        The method assumes each error (except the most probable one for each syndrome)
        contributes to the failure rate based on its number of elements and a given
        probability `p`.

        Updates the attribute `failure_rate` with the calculated value.

        Parameters
        ----------
        p : float, optional
            The probability to be used for the calculation.
        """
        
        failure_rate = 0.
        for _, _, errors in self.syndrome_table_with_leaders:
            # Ignore the most probable error for each syndrome
            missed_errors = errors[1:]
            for error in missed_errors:
                n_errors = len(error.split())
                # Accumulate the contribution of each error to the failure rate
                failure_rate += p**n_errors

        # Update the attribute with the calculated failure rate
        self.failure_rate = failure_rate


    def get_decision_tree(self, measured_syndromes=None, measurements=None):
        """
        Constructs a decision tree for error correction based on measured syndromes and measurements.

        The method selects relevant rows from the syndrome table based on measured syndromes
        and measurements, and constructs a decision tree by finding the stabilizer with
        the highest Shannon entropy at each step.

        The tree is stored as a list of lists, where each list contains a stabilizer and a 
        list of its children.

        Parameters
        ----------
        measured_syndromes : list, optional
            The syndromes that have already been measured.

        measurements : list, optional
            The results of the measurements for the already measured syndromes.
        """
        
        if measured_syndromes is None:
            self.get_syndrome_table()

        # Select relevant rows from the syndrome table based on measurements
        good_rows = self.syndrome_table[:, measured_syndromes] == measurements
        working_table = self.syndrome_table[good_rows]

        decision_tree = []
        measured_stabilizers = []
        n_layers = 1

        # Find the stabilizer with the highest Shannon entropy in the syndrome table
        next_stabilizer = np.argmax(shannon_entropy(self.syndrome_table))

        # Append the stabilizer to the decision tree
        decision_tree.append([next_stabilizer, []])
        measured_stabilizers.append(next_stabilizer)

        # Update the working table by weighting it with noise probabilities, and
        # find the stabilizer with the highest Shannon entropy in the updated table
        next_stabilizer = np.argmax(shannon_entropy(working_table * self.noise_probabilities[good_rows,None]))
        measured_syndromes += [next_stabilizer]

        # Continue building the decision tree recursively
        # self.get_decision_tree(measured_syndromes, measurements+[0])
        # self.get_decision_tree(measured_syndromes, measurements+[1])

def shannon_entropy(probabilities):
    """
    Compute the Shannon entropy of a given set of probabilities.

    Parameters
    ----------
    probabilities : np.array
        Array of probabilities.

    Returns
    -------
    float
        The Shannon entropy value.
    """
    # Use np.where to avoid taking log of 0 which results in nan
    # If probability is 0, treat it as 0; else, use the formula p * log2(p)
    entropy = -np.sum(np.where(probabilities == 0., 0., probabilities * np.log2(probabilities)), axis=0)

    return entropy
        
# GPT-4
def apply_iteration(circuit, qubit_count, stabilizer_count, qubit, stabilizers, max_iterations):
    """
    Applies stabilizers on a quantum circuit iteratively. 

    Parameters
    ----------
    circuit : QuantumCircuit
        The initial quantum circuit.
    qubit_count : int
        Total number of qubits.
    stabilizer_count : int
        Number of stabilizers.
    qubit : QuantumRegister
        Quantum register representing the qubits.
    stabilizers : list
        List of stabilizer operations.
    max_iterations : int
        Maximum number of iterations to perform.

    Returns
    -------
    QuantumCircuit
        The quantum circuit after applying stabilizers.
    """
    ancilla = []
    classical_bit = []
    
    for iteration in range(max_iterations):
        # Add new ancilla and classical bit registers for each iteration
        ancilla.append(QuantumRegister(qubit_count - stabilizer_count, 'a[{}]'.format(iteration)))
        classical_bit.append(ClassicalRegister(qubit_count - stabilizer_count, 'c[{}]'.format(iteration)))
        
        circuit.add_register(ancilla[iteration])
        circuit.add_register(classical_bit[iteration])

        # Apply stabilizers
        circuit = QGRAND.apply_stabilizers(circuit, qubit, ancilla, stabilizers, iteration)

        # Measure the ancilla
        circuit.measure(ancilla[iteration], classical_bit[iteration])

        if iteration != max_iterations - 1:
            # Add barrier for all ancilla and qubits
            circuit.barrier(*ancilla, qubit)

            # Conditionally apply X-gate based on the value of classical bit
            for result in range(1, 2 ** (qubit_count - stabilizer_count)):
                circuit.x(qubit[0]).c_if(classical_bit[iteration], result)
        
    return circuit

def apply_iteration_old(circuit, n, k, qb, stabilizers, max_iter):
    anc, cb = [], []
    for ind in range(max_iter):
        anc.append(QuantumRegister(n-k, 'a[{}]'.format(ind)))
        cb.append(ClassicalRegister(n-k, 'c[{}]'.format(ind)))
        circuit.add_register(anc[ind])
        circuit.add_register(cb[ind])        

        circuit = QGRAND.apply_stabilizers(circuit, qb, anc, stabilizers, ind)

        circuit.measure(anc[ind], cb[ind])

        if ind != max_iter-1:
            circuit.barrier(*anc, qb)
            for res in range(1, 2**(n-k)):
                circuit.x(qb[0]).c_if(cb[ind], res)
        
    return circuit







       
    

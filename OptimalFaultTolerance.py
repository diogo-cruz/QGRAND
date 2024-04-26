from functools import lru_cache
import math
from .FaultTolerance import FaultTolerance
from .GoodCode import GoodCode
from .encoding import get_random_encoding
import numpy as np
from scipy.special import comb
from itertools import combinations, product
from copy import deepcopy
from time import time

class OptimalFaultTolerance(FaultTolerance):
    """
    Class for implementing optimal fault tolerance using quantum error correction.
    It inherits from the FaultTolerance class.
    """

    def get_encoding(self):
        """
        Get the encoding of the quantum error correction code, 
        which is randomly generated using specified parameters.
        """
        # Decompose the encoding circuit twice to ensure it is in its most simplified form.
        self.stabilizers, logical_ops, self.encoding = get_random_encoding(self.n,
                                        self.k,
                                        self.num_gates,
                                        self.num_local_qubits,
                                        self.qr,
                                        seed=self.seed,
                                        stabilizer_format='str',
                                        return_circuit=True,
                                        return_logical_operators=True)
        self.logical_Xs, self.logical_Zs = logical_ops
        self.encoding = self.encoding.decompose().decompose()

    def get_base_errors(self):
        """
        Get base errors in the quantum error correction code, 
        converting error strings to integer arrays for easier processing.
        """
        # Initialize array for base errors
        self.base_errors = np.zeros((self.n_cx, len(self.gate_error_list), 2*self.n), dtype=int)

        # Loop over the error list and convert error to integer arrays
        for i, _ in enumerate(self.base_errors):
            for j, _ in enumerate(self.base_errors[0]):
                error = self.ev_error_list[i][j]
                error = error.lstrip('-')  # Remove leading '-' if present
                error = error[:self.n]

                # Convert error string to int array
                # Error is in [X|Z] format
                array = np.zeros(2*self.n, dtype=int)
                for ind, op in enumerate(reversed(error)):
                    if op == 'I':
                        continue
                    if op != 'Z':
                        array[ind] = 1
                    if op != 'X':
                        array[self.n+ind] = 1

                self.base_errors[i,j] = array

    def syndrome_generator(self, max_t, n_errors):
        """
        Generate syndromes for the quantum error correction code, 
        based on combinations of errors.
        """
        # Array for storing the syndromes
        array = np.zeros(self.n_total-self.n, dtype=int)
        err_array = np.zeros(2*self.n, dtype=int)

        # Loop over all combinations of error operations
        for dd in range(1, max_t+1):
            combos = combinations(range(self.n_cx), dd)
            all_ops = product(range(1, n_errors+1), repeat=dd)
            case = product(all_ops, combos)
            for error_nums, inds in case:
                for ind_err, ind_cx in zip(error_nums, inds):
                    syndrome = self.syndrome_power_set(ind_err, ind_cx)
                    array = (array + syndrome) % 2
                    error = self.error_power_set(ind_err, ind_cx)
                    err_array = (err_array + error) % 2
                yield dd, array, err_array
                array[:] = 0
                err_array[:] = 0

    def get_base_syndromes(self, show=False, change_syndromes=True):
        """
        Get base syndromes for the quantum error correction code, 
        and apply any necessary transformations to the syndromes.
        """
        # Initialize array for base syndromes
        self.base_syndromes = np.zeros((self.n_cx, len(self.gate_error_list), self.n_total-self.n), dtype=int)
        anc = (self.n - self.k)
        s = (self.n_total - self.n)//anc

        # Loop over all possible errors
        for i, ev_errors in enumerate(self.ev_error_list):
            for j, error in enumerate(ev_errors):
                self.base_syndromes[i,j] = self._get_syndrome(error)

                # Change syndromes as required
                if change_syndromes:
                    for si in range(s):
                        self.base_syndromes[i,j][si*anc:(si+1)*anc] = (self.syndrome_change @ self.base_syndromes[i,j][si*anc:(si+1)*anc]) % 2

                # Print the base syndromes if show is True
                if show:
                    print(i, j, error, self.base_syndromes[i,j])

    def error_power_set(self, ind_err, ind_cx):
        """
        Generate a power set of errors for a given index of errors and controlled-X gates.
        """
        errors = self.base_errors[ind_cx]
        array = np.zeros(2*self.n, dtype=int)

        # Loop over the binary representation of the error index
        for i, ind in enumerate(self.bin_rep[ind_err]):
            if ind == 1:
                array = (array + errors[i]) % 2
        return array

    def set_parity_check_matrix(self, row_echelon=True):
        """
        Constructs the parity check matrix and transforms it to row echelon form if specified.
        
        Parameters
        ----------
        row_echelon : bool
            If True (default), transforms the parity check matrix to row echelon form.
        
        Returns
        -------
        None
        """

        # Initialize a zero matrix
        parity_check_matrix = np.zeros((self.n-self.k, 2*self.n), dtype=int)
        
        # Each stabilizer corresponds to a row in the parity check matrix
        for i, stab in enumerate(self.stabilizers):
            stab = list(stab)[1:]
            array = np.zeros(self.n*2, dtype=int)
            
            # Process each element in the stabilizer
            for ind, op in enumerate(reversed(stab)):
                if op == 'I':
                    continue
                if op != 'X':
                    array[self.n+ind] = 1
                if op != 'Z':
                    array[ind] = 1
                    
            parity_check_matrix[i] = array
        self.parity_check_matrix = parity_check_matrix

        # Transform the parity check matrix to row echelon form
        if row_echelon:
            M = parity_check_matrix.copy()
            syndrome_change = np.eye(self.n-self.k, dtype=int)

            lead = 0
            n_rows, n_cols = M.shape
            for r in range(n_rows):
                if lead >= n_cols:
                    break
                i = r
                while M[i, lead] == 0:
                    i += 1
                    if i == n_rows:
                        i = r
                        lead += 1
                        if lead == n_cols:
                            break
                if i != r:
                    M[[i,r]] = M[[r,i]]
                    syndrome_change[[i,r]] = syndrome_change[[r,i]]
                rows_to_change = np.nonzero(M[:, lead])[0]
                rows_to_change = rows_to_change[rows_to_change!=r]
                M[rows_to_change, lead:] = (M[rows_to_change, lead:] + M[r, lead:]) % 2
                syndrome_change[rows_to_change] = (syndrome_change[rows_to_change] + syndrome_change[r]) % 2
                lead += 1

            # Store the row-reduced echelon form matrix, flipping the syndrome change matrix
            self.rre_parity_check_matrix = M
            self.syndrome_change = np.flip(syndrome_change)

            # Find pivot points
            pivots = np.full(M.shape[1], -1, dtype=int)
            shift = 0
            for i in range(M.shape[0]):
                while M[i, i+shift] != 1 and i+shift < M.shape[1]:
                    shift += 1
                if i+shift < M.shape[1]:
                    pivots[i+shift] = i
            self.pivots = pivots

    def set_logical_operators(self):
        """
        Sets up logical operators for a quantum error correction code.
        
        This function generates logical matrices Xs and Zs for quantum error correction codes 
        using the given stabilizers. Then it generates a reduced row echelon form (rref) of the logical matrix,
        and finds the pivot positions in this rref matrix.

        Notes
        -----
        The logical matrix is in [X|Z] format, hence the size of the logical matrix is 2*k x 2*n.

        """
        def generate_logical_array(stabilizers):
            """
            Generate a logical array from the given stabilizers.

            Parameters
            ----------
            stabilizers : list of str
                List of stabilizer strings where each string represents a Pauli operator acting on qubits.

            Returns
            -------
            logical_array : np.ndarray
                Logical array generated from the stabilizers.
            """
            logical_array = np.zeros((len(stabilizers), self.n*2), dtype=int)
            for i, stab in enumerate(stabilizers):
                stab = list(stab)[1:]  # Remove the first character as it's the operator type
                for ind, op in enumerate(reversed(stab)):
                    if op == 'I':  # If operator is identity, continue without change
                        continue
                    if op != 'X':  # If operator is not X, set corresponding Z position to 1
                        logical_array[i, self.n+ind] = 1
                    if op != 'Z':  # If operator is not Z, set corresponding X position to 1
                        logical_array[i, ind] = 1
            return logical_array

        # Generate logical arrays from X and Z stabilizers
        logical_matrix = np.vstack((generate_logical_array(self.logical_Xs),
                                    generate_logical_array(self.logical_Zs)))

        self.logical_matrix = logical_matrix

        # Generate reduced row echelon form (rref) of logical matrix
        self.rre_logical_matrix = logical_matrix.copy()
        col_size = self.rre_parity_check_matrix.shape[1]
        for i in range(col_size):
            ind = self.pivots[i]
            if ind != -1:
                rows_to_change = np.nonzero(self.rre_logical_matrix[:, i])[0]
                self.rre_logical_matrix[rows_to_change, i:] = (self.rre_logical_matrix[rows_to_change, i:] + self.rre_parity_check_matrix[ind, i:]) % 2

        # Bring the rref matrix to canonical form by swapping rows when needed
        M = self.rre_logical_matrix
        n_rows, n_cols = M.shape
        lead = 0  # Initialize the leading position
        for r in range(n_rows):
            if lead >= n_cols:
                break
            i = r
            while M[i, lead] == 0:
                i += 1
                if i == n_rows:
                    i = r
                    lead += 1
                    if lead == n_cols:
                        break
            if i != r:
                M[[i, r]] = M[[r, i]]  # Swap rows i and r
            rows_to_change = np.nonzero(M[:, lead])[0]
            rows_to_change = rows_to_change[rows_to_change != r]
            M[rows_to_change, lead:] = (M[rows_to_change, lead:] + M[r, lead:]) % 2
            lead += 1

        self.rre_logical_matrix = M

        # Find pivot positions in the rref logical matrix
        pivots = np.full(M.shape[1], -1, dtype=int)
        shift = 0
        for i in range(M.shape[0]):
            while M[i, i+shift] != 1 and i+shift < M.shape[1]:
                shift += 1
            if i+shift < M.shape[1]:
                pivots[i+shift] = i

        self.logical_pivots = pivots

    def get_weight_statistics(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            truncate=True,
                            consider_degeneracy=False,
                            show=False):
        """
        This method computes weight statistics based on error syndromes.
        
        Parameters
        ----------
        prob : float or np.ndarray, optional
            The probability of an error, either as a single value or as an array of probabilities.
        max_t : int, optional
            The maximum error weight. If not provided, it defaults to the number of CNOT operations.
        fill_syndromes : bool, optional
            Whether to initialize all possible syndrome values. Defaults to False.
        n_cx_errors : int, optional
            Number of possible CNOT errors. Defaults to 15.
        truncate : bool, optional
            Whether to limit the error weight to max_t. Defaults to True.
        consider_degeneracy : bool, optional
            Whether to consider degenerate errors (different errors that lead to the same syndrome). Defaults to False.
        show : bool, optional
            Whether to print the intermediate results. Defaults to False.
            
        Returns
        -------
        tuple
            Returns failure rate, correctable fraction and mean iterations, either for non-degenerate case only or
            for both non-degenerate and degenerate cases.
        """

        # Initialize required values
        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        # Set max_t to the number of CNOT operations if not specified
        max_t = max_t if max_t is not None else n_cx
        n_show = max_t if truncate else n_cx

        # Ensure 'prob' is a numpy array for further calculations
        prob = np.array(prob if isinstance(prob, float) else prob)

        # Initialize dictionary for syndrome-error mapping
        syndrome_error_dict = {0:0}
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        # If degeneracy is considered, initialize respective arrays
        if consider_degeneracy:
            error_type_degen = np.zeros(n_show, dtype=int)
            probabilities_degen = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])

        fail_size = int(sum(n_cx_errors**i * comb(n_cx, i) for i in range(1, max_t+1)))
        fail_arr = np.zeros(fail_size)
        fail = 0

        # Iterate through generated syndromes
        for i, (error_weight, syndrome, error) in enumerate(self.syndrome_generator(max_t, n_cx_errors)):

            int_syndrome = self.bin_array_to_int(syndrome)
            int_error = self.bin_array_to_int(error)

            # Check if the syndrome is new
            if int_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_syndrome] = int_error
                error_type[error_weight-1] += 1

            # Check if error occurred before
            elif int_error == syndrome_error_dict[int_syndrome]:
                error_type[error_weight-1] += 1

            # Check for degenerate error
            elif consider_degeneracy and self.error_is_degenerate(int_error, syndrome_error_dict[int_syndrome]):
                error_type_degen[error_weight-1] += 1

            # Track the number of unhandled cases
            else:
                fail += 1

            # Print failure rate for every 100000 syndromes
            if i%100000==0:
                print("Fail {}: {}".format(i//100000, fail/100000))
                fail_arr[i//100000] = fail/100000
                fail = 0

        t_arr = 1 + np.arange(max_t)[:, None]
        probabilities = error_type[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)

        # If degeneracy was considered, calculate respective probabilities
        if consider_degeneracy:
            probabilities_degen = error_type_degen[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)

        failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

        n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
        correctable_fraction = error_type / n_errors

        # Calculate mean iterations
        mean_iterations = np.array([
            sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                    sum(error_type[:dd]) + (nn+1)/2
                ) for dd, nn in enumerate(error_type[:i])
            ) for i in range(1,n_show+1)
        ])

        if consider_degeneracy:
            error_type += error_type_degen
            probabilities += probabilities_degen

            failure_rate_all = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

            correctable_fraction_all = error_type / n_errors

            mean_iterations_all = np.array([
                sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                        sum(error_type[:dd]) + (nn+1)/2
                    ) for dd, nn in enumerate(error_type[:i])
                ) for i in range(1,n_show+1)
            ])

        # Return statistics either only for non-degenerate case or for both non-degenerate and degenerate cases
        return (failure_rate, failure_rate_all), (correctable_fraction, correctable_fraction_all), (mean_iterations, mean_iterations_all) if consider_degeneracy else (failure_rate, correctable_fraction, mean_iterations)

    def get_pure_errors_ind(self):
        """
        Compute pure errors indices.

        Sets an attribute `pure_errors_ind` with reverse pivots indices.
        The indices are calculated based on the number of errors and the value of pivots.

        Notes
        -----
        `self.n`, `self.k`, and `self.pivots` should be defined before calling this method.
        """

        self.pure_errors_ind = np.zeros(self.n-self.k, dtype=int)

        rev_pivots = self.n-self.k-1-self.pivots
        rev_pivots[rev_pivots == self.n-self.k] = -1
        for i, pivot in enumerate(rev_pivots):
            if pivot != -1:
                self.pure_errors_ind[pivot] = i + self.n * (1 if i < self.n else -1)

    def get_zero_stabilizer_compound_error(self, syndrome, error):
        """
        Calculate the stabilizer compound error.

        Parameters
        ----------
        syndrome: ndarray
            The syndrome of the error.
        error: ndarray
            The error that occurred.

        Returns
        -------
        ndarray
            Zero stabilizer compound error.
        """

        # Convert syndrome to boolean type and get the relevant part
        syndrome = syndrome[-(self.n-self.k):].astype(bool)

        # Error is in [X|Z] format
        # Create a copy of the error and modify it based on pure_errors_ind and the syndrome
        zero_error = error.copy()
        zero_error[self.pure_errors_ind[syndrome]] ^= 1

        return zero_error

    def get_degenerate_set(self, zero_error):
        """
        Compute the degenerate set for the given zero_error.

        Parameters
        ----------
        zero_error: ndarray
            Zero stabilizer compound error.

        Returns
        -------
        int
            Integer representation of the degenerate set.
        """
        # Initialize index i and get column size
        i = 0
        col_size = self.rre_parity_check_matrix.shape[1]

        # Go through each column
        while i < col_size:
            if zero_error[i] == 1:
                ind = self.pivots[i]
                if ind != -1:
                    zero_error = (zero_error + self.rre_parity_check_matrix[ind]) % 2
            i += 1

        degenerate_set = np.zeros(2*self.k, dtype=int)
        i = 0
        while i < col_size:
            if zero_error[i] == 1:
                ind = self.logical_pivots[i]
                if ind != -1:
                    degenerate_set[ind] = 1
                    zero_error = (zero_error + self.rre_logical_matrix[ind]) % 2
            i += 1

        assert np.sum(zero_error) == 0

        # Select the appropriate function to convert the binary array to an integer
        bin_array_to_int_func = self.bin_array_to_int_fast if degenerate_set.shape[0] <= 64 else self.bin_array_to_int

        # Convert the binary array to an integer
        degen_int = bin_array_to_int_func(degenerate_set)

        return degen_int

    def get_degen_set(self, syndrome, error):
        """
        Compute the degenerate set for the given syndrome and error.

        Parameters
        ----------
        syndrome: ndarray
            The syndrome of the error.
        error: ndarray
            The error that occurred.

        Returns
        -------
        int
            Integer representation of the degenerate set.
        """
        return self.get_degenerate_set(self.get_zero_stabilizer_compound_error(syndrome, error))
        
    def get_weight_statistics(self, prob=None, max_t=None, fill_syndromes=False,
                                  n_cx_errors=15, truncate=True, show=False):
        """
        Compute weight statistics based on the given parameters.

        Parameters
        ----------
        prob : float or list or np.ndarray, optional
            The probabilities to be used, if it's a float, it will be converted to a list.
        max_t : int, optional
            The maximum time steps to be considered. If None, set to n_cx.
        fill_syndromes : bool, optional
            This flag is actually not used in the function and can be removed.
        n_cx_errors : int, default is 15
            The number of cx errors to consider.
        truncate : bool, default is True
            If True, n_show is set to max_t. If False, n_show is set to n_cx.
        show : bool, optional
            This flag is actually not used in the function and can be removed.

        Returns
        -------
        failure_rate_all : np.ndarray
            The failure rates for the given parameters.

        Notes
        -----
        This function provides a lower bound for the performance of the code.
        It assumes that errors are distinct, therefore it disregards possible 
        degeneracies in the main n qubits, but also the highly degenerate errors 
        in the ancilla qubits.
        """

        # Get the required attributes from the object
        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        # If max_t is not defined, set it as n_cx
        max_t = n_cx if max_t is None else max_t

        # Determine the number to show based on truncate flag
        n_show = max_t if truncate else n_cx

        # Decide which bin_array_to_int function to use based on the total number of qubits
        bin_array_to_int_syndrome = self.bin_array_to_int_fast if n_total-n <= 64 else self.bin_array_to_int
        bin_array_to_int_error = self.bin_array_to_int_fast if 2*n <= 64 else self.bin_array_to_int

        # Ensure the probabilities are in numpy array format
        if isinstance(prob, float):
            prob = np.array([prob])
        elif not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        # Initialize variables
        syndrome_error_dict = {}
        zero_array = np.zeros(max_t+1, dtype=int)
        zero_array[0] = 1
        one_array = np.zeros(max_t+1, dtype=int)
        one_array[1] = 1
        nothing_array = np.zeros(max_t, dtype=int)
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))
        probability_no_error = np.array([(1-p)**n_cx for p in prob])
        fail_size = int(sum(n_cx_errors**i * comb(n_cx, i) for i in range(1, max_t+1)))
        box_size = 10_000
        fail_arr = np.zeros(fail_size//box_size + 1)
        fail = 0
        loops = np.zeros(3, dtype=int)

        # Iterate through the syndromes generated
        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(max_t, n_cx_errors)):
            int_syndrome = bin_array_to_int_syndrome(syndrome)
            degen_int = self.get_degen_set(syndrome, error)

            if int_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_syndrome] = {degen_int: nothing_array.copy()}
                syndrome_error_dict[int_syndrome][degen_int][dd-1] += 1
                case = 0
            elif degen_int in syndrome_error_dict[int_syndrome]:
                syndrome_error_dict[int_syndrome][degen_int][dd-1] += 1
                case = 1
            else:
                syndrome_error_dict[int_syndrome][degen_int] = nothing_array.copy()
                syndrome_error_dict[int_syndrome][degen_int][dd-1] += 1
                case = 2

            loops[case] += 1

        self.loops = loops

        t_arr = 1 + np.arange(max_t)[:, None]
        t_arr_0 = np.arange(max_t+1)[:, None]

        probabilities = np.zeros((max_t, prob.shape[0]))
        for int_syndrome, degen_sets in syndrome_error_dict.items():
            if int_syndrome==0:
                degen_probs = np.zeros((len(degen_sets), max_t+1, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    if degen_int == 0:
                        degen_set = np.insert(degen_set, 0, 1)
                    else:
                        degen_set = np.insert(degen_set, 0, 0)
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr_0 * (1-prob)**(n_cx-t_arr_0)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)[1:]
                probabilities += optimal_prob
            else:
                degen_probs = np.zeros((len(degen_sets), max_t, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)
                probabilities += optimal_prob

        # Calculate the failure rates
        failure_rate_all = 1 - probabilities

        self.fail_arr = fail_arr

        return failure_rate_all
    
    def get_weight_statistics_fast(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            truncate=True,
                            timeout=None,
                            memory_limit=None,
                            fake_meas=False,
                            show=False):
        """
        This function calculates the failure rate of a quantum error correction code by simulating 
        the propagation of errors through the code's gates.

        Parameters
        ----------
        prob : float or list
            Error probabilities for each gate.
        max_t : int, optional
            Maximum time for which to calculate the failure rate. If None, the number of gates is used.
        fill_syndromes : bool, optional
            If True, fill the syndromes with errors, default is False.
        n_cx_errors : int, optional
            The number of times a control gate can error, default is 15.
        truncate : bool, optional
            If True, output is truncated to max_t, default is True.
        timeout : float, optional
            Time limit in seconds, default is None.
        memory_limit : int, optional
            Maximum number of elements in the syndrome dictionary, default is None.
        fake_meas : bool, optional
            If True, adds fake measurements to the end of the circuit, default is False.
        show : bool, optional
            If True, the function prints out progress reports, default is False.

        Returns
        -------
        failure_rate_all : numpy array
            Failure rates for each time step and each error probability.
        """
        # Starting time of function execution
        start_time = time()

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total
        if fake_meas:
            self.n_meas = n_total - n - self._ignore_extra_stabs * (n - k)
            n_cx += self.n_meas
        
        # If max_t is not provided, use the number of gates as default
        max_t = n_cx if max_t is None else max_t

        # Number of probabilities to show is determined by whether truncation is applied
        n_show = max_t if truncate else n_cx

        # Choose binary to integer conversion function depending on the size
        bin_array_to_int_syndrome = self.bin_array_to_int_fast if n_total - n <= 64 else self.bin_array_to_int

        # Convert probabilities to numpy array
        if isinstance(prob, float):
            prob = np.array([prob])
        elif not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        # Initialize syndrome error dictionary and other variables
        syndrome_error_dict = {}
        nothing_array = np.zeros(max_t, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        # Step 1: Generate the syndromes
        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(1, n_cx_errors)):

            int_syndrome = bin_array_to_int_syndrome(syndrome)
            degen_int = self.get_degen_set(syndrome, error)

            if int_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_syndrome] = {degen_int:nothing_array.copy()}

            elif degen_int not in syndrome_error_dict[int_syndrome]:
                syndrome_error_dict[int_syndrome][degen_int] = nothing_array.copy()
            
            syndrome_error_dict[int_syndrome][degen_int][dd-1] += 1

        # Step 2: Combine the syndromes
        syndrome_error_dict_1 = deepcopy(syndrome_error_dict)
        len_dict_1 = len(syndrome_error_dict_1)
        print(n_cx_errors*n_cx, len_dict_1, len_dict_1**2)
        broke_out = False
        for dd in range(2, max_t+1):
            for int_syndrome_big in list(syndrome_error_dict):
                degen_sets_big = syndrome_error_dict[int_syndrome_big]
                for int_degen_big in list(degen_sets_big):
                    data_2 = degen_sets_big[int_degen_big]
                    for int_syndrome_1, degen_sets_1 in syndrome_error_dict_1.items():
                        for int_degen_1, data_1 in degen_sets_1.items():

                            int_syndrome = int_syndrome_1 ^ int_syndrome_big
                            int_degen = int_degen_1 ^ int_degen_big

                            if int_syndrome not in syndrome_error_dict:
                                syndrome_error_dict[int_syndrome] = {int_degen:nothing_array.copy()}

                            elif int_degen not in syndrome_error_dict[int_syndrome]:
                                syndrome_error_dict[int_syndrome][int_degen] = nothing_array.copy()
                            
                            syndrome_error_dict[int_syndrome][int_degen][dd-1] += data_1[0] * data_2[dd-2]

        
                if timeout is not None:
                    end_time = time()
                    if end_time - start_time > timeout:
                        print("Timeout!")
                        broke_out = True
                        break
                if memory_limit is not None:
                    if len(syndrome_error_dict) > memory_limit:
                        print("Out of memory!")
                        broke_out = True
                        break

            if dd>2:
                for int_syndrome in list(syndrome_error_dict):
                    degen_sets = syndrome_error_dict[int_syndrome]
                    for degen_int, degen_set in degen_sets.items():
                        a = degen_set[dd-1] - degen_set[dd-3]*n_cx_errors*(n_cx - (dd-2)) - (n_cx_errors-1)*(dd-1)*degen_set[dd-2]
                        #assert a%dd==0
                        syndrome_error_dict[int_syndrome][degen_int][dd-1] = a // dd
            else:
                for int_syndrome in list(syndrome_error_dict):
                    degen_sets = syndrome_error_dict[int_syndrome]
                    for degen_int, degen_set in degen_sets.items():
                        if int_syndrome==0 and degen_int==0:
                            a = degen_set[dd-1] - n_cx_errors*(n_cx - (dd-2)) - (n_cx_errors-1)*(dd-1)*degen_set[dd-2]
                            #assert a%dd==0
                            syndrome_error_dict[int_syndrome][degen_int][dd-1] = a // dd
                        else:
                            a = degen_set[dd-1] - (n_cx_errors-1)*(dd-1)*degen_set[dd-2]
                            #assert a%dd==0
                            syndrome_error_dict[int_syndrome][degen_int][dd-1] = a // dd

            print(dd, len(syndrome_error_dict), len_dict_1*len(syndrome_error_dict))

            if broke_out:
                break

        # Step 3: Calculate the failure rates
        t_arr = 1 + np.arange(max_t)[:, None]
        t_arr_0 = np.arange(max_t+1)[:, None]

        probabilities = np.zeros((max_t, prob.shape[0]))
        for int_syndrome, degen_sets in syndrome_error_dict.items():
            if int_syndrome==0:
                degen_probs = np.zeros((len(degen_sets), max_t+1, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    if degen_int == 0:
                        degen_set = np.insert(degen_set, 0, 1)
                    else:
                        degen_set = np.insert(degen_set, 0, 0)
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr_0 * (1-prob)**(n_cx-t_arr_0)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)[1:]
                probabilities += optimal_prob
            else:
                degen_probs = np.zeros((len(degen_sets), max_t, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)
                probabilities += optimal_prob

            failure_rate_all = 1 - probabilities

        # Assign the final syndrome error dictionary to self.syndrome_error_dict
        self.syndrome_error_dict = dict(sorted(syndrome_error_dict.items()))

        # Return the failure rates
        return failure_rate_all

    def get_weight_statistics_fast_optimized(self, prob=None, max_t=None, n_cx_errors=15, truncate=True):
        """
        Calculate the failure rate for weight statistics.

        Parameters
        ----------
        prob: float, numpy.ndarray or None, optional
            Error probability or probabilities to consider. If it's None, the default will be used.
        max_t: int or None, optional
            Maximum number of errors to consider. If it's None, the number of controlled-X gates will be used as the maximum.
        n_cx_errors: int, optional
            The number of errors in controlled-X gates.
        truncate: bool, optional
            Whether to truncate the output. If true, the output will be limited to `max_t` errors.

        Returns
        -------
        numpy.ndarray
            Failure rates for each error probability.
        """
        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        # If max_t is not defined, consider up to the number of controlled-X gates errors
        max_t = n_cx if max_t is None else max_t

        n_show = max_t if truncate else n_cx

        bin_array_to_int_syndrome = self.bin_array_to_int_fast if n_total-n <= 64 else self.bin_array_to_int

        # Make sure prob is a numpy array
        if isinstance(prob, float):
            prob = np.array([prob])
        elif not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        syndrome_error_dict = {}
        degen_nothing_array = np.zeros((2**(2*k), max_t), dtype=int)

        # Iterate over all generated syndromes
        for _, syndrome, error in self.syndrome_generator(1, n_cx_errors):
            int_syndrome = bin_array_to_int_syndrome(syndrome)
            degen_int = self.get_degen_set(syndrome, error)

            # Add the syndrome to the dictionary if not already present
            if int_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_syndrome] = degen_nothing_array.copy()

            syndrome_error_dict[int_syndrome][degen_int, 0] += 1

        #syndrome_error_dict_1 = syndrome_error_dict.copy()
        syndrome_error_dict_1 = deepcopy(syndrome_error_dict)
        for dd in range(2, max_t+1):
            syndrome_error_dict_old = deepcopy(syndrome_error_dict)
            for int_syndrome_big, degen_sets_big in syndrome_error_dict_old.items():
                for int_degen_big, data_2 in enumerate(degen_sets_big):
                    for int_syndrome_1, degen_sets_1 in syndrome_error_dict_1.items():
                        for int_degen_1, data_1 in enumerate(degen_sets_1):

                            int_syndrome = int_syndrome_1 ^ int_syndrome_big
                            int_degen = int_degen_1 ^ int_degen_big

                            if int_syndrome not in syndrome_error_dict:
                                syndrome_error_dict[int_syndrome] = degen_nothing_array.copy()

                            syndrome_error_dict[int_syndrome][int_degen, dd-1] += data_1[0] * data_2[dd-2]

            syndrome_error_dict_old = deepcopy(syndrome_error_dict)

            # Modify the syndrome error dict based on calculated degen_array
            for int_syndrome, degen_array in syndrome_error_dict.items():
                if dd > 2:
                    a = degen_array[:,dd-1] - degen_array[:,dd-3]*n_cx_errors*(n_cx - (dd-2)) - (n_cx_errors-1)*(dd-1)*degen_array[:,dd-2]
                else:
                    a = degen_array[:,dd-1] - (n_cx_errors-1)*(dd-1)*degen_array[:,dd-2]
                    if int_syndrome == 0:
                        a[0] -= n_cx_errors*(n_cx - (dd-2))
                syndrome_error_dict[int_syndrome][:,dd-1] = a // dd

        t_arr = 1 + np.arange(max_t)[None, :, None]
        t_arr_0 = np.arange(max_t+1)[None, :, None]

        probabilities = np.zeros((max_t, prob.shape[0]))
        for int_syndrome, degen_array in syndrome_error_dict.items():
            if int_syndrome == 0:
                degen_array = np.insert(degen_array, 0, 0, axis=1)
                degen_array[0,0] = 1
                degen_probs = degen_array[:,:,None] * (prob/n_cx_errors)**t_arr_0 * (1-prob)**(n_cx-t_arr_0)
            else:
                degen_probs = degen_array[:,:, None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)

            degen_probs = np.cumsum(degen_probs, axis=1)
            optimal_prob = np.amax(degen_probs, axis=0)

            # ADDED BY DIOGO
            if int_syndrome == 0:
                optimal_prob = optimal_prob[1:]

            probabilities += optimal_prob

        failure_rate_all = 1 - probabilities

        return failure_rate_all

    def get_weight_statistics_faster(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            truncate=True,
                            show=False):
        r"""Computes decoding table in logarithmically many steps.

        The procedure for :math:`t=1` is exactly as in 
        :py:func:`get_weight_statistics`. Once we have the data dictionary for 
        :math:`t=1`, we can start to optimize.

        Instead of iterating through compound errors individually, we iterate 
        through the degenerate sets obtained in the :math:`t=1` step. So if we 
        had the dictionary

        .. math::
            \{\mathbf s_1:\{\mathcal D_{11}:[0,N_{11},0,\ldots], 
            \mathcal D_{12}:[0,N_{12},0,\ldots],\ldots\}, 
            \mathbf s_2:\{\mathcal D_{21}:[0,N_{21},0,\ldots], 
            \mathcal D_{22}:[0,N_{22},0,\ldots],\ldots\}\},

        instead of iterating through the combinations 
        :math:`\{E_1E_2, E_1E_3,\ldots, E_2E_3,\ldots\}` as before, we now 
        iterate through the degenerate sets :math:`\mathcal D_{ij}` as a whole.

        Consider the data dictionary :math:`\mathcal D_{t-1}` that includes the 
        errors up to weight :math:`t-1`. To obtain the dictionary for :math:`t`,
        we iterate through the combination of the degenerate sets in 
        :math:`\mathcal D_{t-1}` and :math:`\mathcal D_1`. If the noise 
        statistics are highly degenerate, we can have considerable 
        computational savings, since we only need to perform 
        :math:`|\mathcal D_{t-1}||\mathcal D_1|` computations instead of 
        :math:`15^{t}\begin{pmatrix}N_{\rm CNOT} \\ t\end{pmatrix}`. While we 
        expect the latter to grow quickly with :math:`\mathcal O(n^{2t})`, the 
        former approach should grow, at worst, with :math:`\mathcal O(n^t)`.

        With this approach we might overcount the number of error patterns. 
        There will be to types of overcounting:

        * Counting permuted copies: Consider a weight-:math:`(t-1)` 
          pattern :math:`E_{i_1}E_{i_2}\ldots E_{i_{t-1}}` (with 
          :math:`i_1 < i_2 < i_3 < \ldots`), coming from 
          :math:`\mathcal{D}_{t-1}`, and the error :math:`E_j`, coming from 
          :math:`\mathcal D_1`. W.l.o.g., suppose :math:`j < i_1`. Then, for 
          :math:`\mathcal D_t`, we will not only count the pattern 
          :math:`E_j E_{i_1}E_{i_2}\ldots E_{i_{t-1}}` coming from here, but 
          also that same pattern coming from the combination of the patterns 
          :math:`E_j E_{i_1}E_{i_2}\ldots E_{i_{t-1}} \ E_{i_k}` and 
          :math:`E_{i_k}`. In total, we overcount each weight-:math:`t` error 
          :math:`t` times.
        
        * Recounting lower weight errors: for the error pattern 
          :math:`E_{i_1}E_{i_2}\ldots E_{i_{t-1}}`, composing with any 
          :math:`E_{i_k}` (:math:`1\leq k\leq t-1`) reduces the error pattern 
          to one with weight :math:`t-2`, which was previously counted. Each 
          weight-:math:`(t-2)` error we counted before will be encountered 
          here :math:`\zeta` times.

        Given these factors, if our iterative process yields an 
        :math:`\tilde N_t` count for some degenerate set on 
        :math:`\mathcal D_t`, then the right :math:`N_t` count is related to 
        this by
        
        .. math::
            N_t &= \frac{\tilde N_t - \sum_{\substack{k+p\leq b\\ k,p\geq 0}} 
                                    \zeta_{a,b}(k,p) N_{a+b-2k-p}}{R_{a,b}}\\
            R_{a,b} &:= \frac{
                \begin{pmatrix}N_{\rm CNOT} \\ a\end{pmatrix}
                \begin{pmatrix}N_{\rm CNOT}-a \\ b\end{pmatrix}
                }{\begin{pmatrix}N_{\rm CNOT} \\ a+b\end{pmatrix}}\\
            \zeta_{a,b}(k,p) &:= C^k (C-1)^p\frac{
                \begin{pmatrix}N_{\rm CNOT} \\ a\end{pmatrix}
                \begin{pmatrix}N_{\rm CNOT}-a \\ b-k-p\end{pmatrix}
                \begin{pmatrix}a\\ k\end{pmatrix}
                \begin{pmatrix}a-k\\ p\end{pmatrix}
                }{\begin{pmatrix}N_{\rm CNOT} \\ a+b-2k-p\end{pmatrix}}.

        That's it.
        """

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        # Check is max_t is power of two
        assert (max_t & (max_t-1) == 0) and (max_t != 0)

        if max_t is None:
            max_t = n_cx

        if truncate:
            n_show = max_t
        else:
            n_show = n_cx

        if n_total-n <= 64:
            bin_array_to_int_syndrome = self.bin_array_to_int_fast
        else:
            bin_array_to_int_syndrome = self.bin_array_to_int

        # Converting prob to numpy array
        if isinstance(prob, float):
            prob = [prob]

        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        syndrome_error_dict = {}
        nothing_array = np.zeros(max_t, dtype=int)

        probabilities = np.zeros((n_show, prob.shape[0]))

        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(1, n_cx_errors)):

            int_syndrome = bin_array_to_int_syndrome(syndrome)
            degen_int = self.get_degen_set(syndrome, error)

            if int_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_syndrome] = {degen_int:nothing_array.copy()}

            elif degen_int not in syndrome_error_dict[int_syndrome]:
                syndrome_error_dict[int_syndrome][degen_int] = nothing_array.copy()
            
            syndrome_error_dict[int_syndrome][degen_int][dd-1] += 1

        #syndrome_error_dict_1 = deepcopy(syndrome_error_dict)
        #print(n_cx_errors*n_cx, len(syndrome_error_dict_1), len(syndrome_error_dict_1)**2)

        log_max_t = int(round(math.log(max_t, 2), 0))

        start_time = time()

        for iter in range(log_max_t):
            size = 2**iter
            #dd = 2*size
            old_syndrome_keys = list(syndrome_error_dict)
            for int_syndrome_big in old_syndrome_keys:
                degen_sets_big = syndrome_error_dict[int_syndrome_big]
                for int_degen_big in list(degen_sets_big):
                    data_2 = degen_sets_big[int_degen_big]
                    for int_syndrome_1 in old_syndrome_keys:
                        degen_sets_1 = syndrome_error_dict[int_syndrome_1]
                        for int_degen_1 in list(degen_sets_1):
                            data_1 = degen_sets_1[int_degen_1]

                            int_syndrome = int_syndrome_1 ^ int_syndrome_big
                            int_degen = int_degen_1 ^ int_degen_big

                            if int_syndrome not in syndrome_error_dict:
                                syndrome_error_dict[int_syndrome] = {int_degen:nothing_array.copy()}

                            elif int_degen not in syndrome_error_dict[int_syndrome]:
                                syndrome_error_dict[int_syndrome][int_degen] = nothing_array.copy()
                            
                            syndrome_error_dict[int_syndrome][int_degen][size:2*size] += data_1[size-1] * data_2[:size]

            middle_time = time()

            for int_syndrome, degen_sets in syndrome_error_dict.items():
                #degen_sets = syndrome_error_dict[int_syndrome]
                for degen_int, degen_set in degen_sets.items():
                    a = size
                    for dd in range(size+1, 2*size+1):
                        b = dd - a
                        total_overcount = 0
                        for k in range(b+1):
                            for p in range(b-k+1):
                                if k==p==0:
                                    continue
                                c = a + b - 2*k - p
                                if c==0:
                                    if int_syndrome==0 and degen_int==0:
                                        count = 1
                                    else:
                                        continue
                                else:
                                    count = degen_set[c-1]
                                
                                total_overcount += count * self.overcounted(a,b,k,p,c,n_cx_errors,n_cx)
                        num = degen_set[dd-1] - total_overcount
                        den = comb(n_cx,a,exact=True)*comb(n_cx-a,b,exact=True)//comb(n_cx,a+b,exact=True)
                        assert num%den==0
                        syndrome_error_dict[int_syndrome][degen_int][dd-1] = num // den

            end_time = time()
            print(middle_time-start_time, end_time-middle_time)


            #print(dd, len(syndrome_error_dict), len(syndrome_error_dict_1)*len(syndrome_error_dict))

        t_arr = 1 + np.arange(max_t)[:, None]
        t_arr_0 = np.arange(max_t+1)[:, None]

        probabilities = np.zeros((max_t, prob.shape[0]))
        for int_syndrome, degen_sets in syndrome_error_dict.items():
            if int_syndrome==0:
                degen_probs = np.zeros((len(degen_sets), max_t+1, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    if degen_int == 0:
                        degen_set = np.insert(degen_set, 0, 1)
                    else:
                        degen_set = np.insert(degen_set, 0, 0)
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr_0 * (1-prob)**(n_cx-t_arr_0)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)[1:]
                probabilities += optimal_prob
            else:
                degen_probs = np.zeros((len(degen_sets), max_t, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)
                probabilities += optimal_prob

            failure_rate_all = 1 - probabilities

        self.syndrome_error_dict = dict(sorted(syndrome_error_dict.items()))

        return failure_rate_all

    @staticmethod
    @lru_cache(maxsize=None)
    def overcounted(a, b, k, p, c, C, N):
        """
        This method calculates the overcounted number based on given parameters.

        Parameters
        ----------
        a, b, k, p, c : int
            Numeric parameters for calculation.
        C, N : int
            Constants for calculation.

        Returns
        -------
        int
            Result of the calculation.
        """
        # Calculate numerator and denominator of the formula
        num = C**k * (C-1)**p * comb(N, a, exact=True) * comb(N-a, b-k-p, exact=True) * comb(a, k, exact=True) * comb(a-k, p, exact=True)
        den = comb(N, c, exact=True)
        
        # Return the integer division of num by den
        return num // den

    @staticmethod
    def just_first(array):
        """
        This method sets the first non-zero element of the array to 1 and others to 0.

        Parameters
        ----------
        array : ndarray
            The input array.

        Returns
        -------
        ndarray
            The transformed array.
        """
        # Create a new array filled with zeros, having the same shape and type as the original array
        new_array = np.zeros_like(array)

        # Find the first non-zero element and set it to 1
        ind = np.nonzero(array)[0][0]
        new_array[ind] = 1

        return new_array

    def error_is_degenerate(self, int_error_1, int_error_2):
        """
        Check if error is degenerate.

        Parameters
        ----------
        int_error_1, int_error_2 : int
            Integer representation of errors.

        Returns
        -------
        bool
            Result of degeneracy check.
        """
        # Convert integer errors to binary representation
        error_1 = np.array(list(np.binary_repr(int_error_1).zfill(self.n*2))).astype(int)
        error_2 = np.array(list(np.binary_repr(int_error_2).zfill(self.n*2))).astype(int)

        # Sum the binary errors and apply mod 2
        error = (error_1 + error_2) % 2

        # Check if error is degenerate
        result = GoodCode._get_boolean_solution_3(self.rre_parity_check_matrix, self.pivots, error)

        return result

    @staticmethod
    def bin_array_to_int(array):
        """
        Convert binary array to integer.

        Parameters
        ----------
        array : ndarray
            The binary array.

        Returns
        -------
        int
            Integer representation of binary array.
        """
        result = 0
        for bit in array:
            result = (result << 1) | bool(bit)
        return result

    @staticmethod
    def bin_array_to_int_fast(array):
        """
        Faster method to convert binary array to integer.

        Parameters
        ----------
        array : ndarray
            The binary array.

        Returns
        -------
        int
            Integer representation of binary array.
        """
        result = 0
        for bit in array.astype(bool):
            result = (result << 1) | bit
        return result

    @staticmethod
    def join_probs(data_1, data_all):
        """
        This method combines two probability arrays.

        Parameters
        ----------
        data_1, data_all : ndarray
            The probability arrays to be combined.

        Returns
        -------
        ndarray
            The combined probability array.
        """
        data = np.roll(data_1[1] * data_all, 1)
        if data_1[0] != 0:
            data += data_1[0] * data_all
        return data

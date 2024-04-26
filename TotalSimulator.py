from .RealisticSimulator import RealisticSimulator
import numpy as np
from scipy.special import comb
from copy import deepcopy
from time import time
from .pauli import pauli_generator_num
from itertools import combinations, product

class TotalSimulator(RealisticSimulator):

    def propagated_syndrome_generator(self, change_syndromes, qe_list, max_te, s):
        """
        Generate propagated syndromes for a quantum error correcting code.

        Parameters
        ----------
        change_syndromes : bool
            If True, use the RRE parity check matrix to generate syndromes. Otherwise, use the standard parity check matrix.
        qe_list : list of int
            List of integers representing the boundaries between blocks of qubits. Each block is treated independently when generating syndromes.
        max_te : int
            The maximum number of errors to consider when generating syndromes.
        s : int
            The number of blocks of qubits.

        Yields
        ------
        tuple of (int, numpy.ndarray, numpy.ndarray, set)
            A tuple containing the current error weight, the full syndrome (concatenated over all qubit blocks), the syndrome for the final block of qubits, and the set of degenerate error patterns for the final block of qubits.
        """


        if change_syndromes:
            parity_check_matrix = self.rre_parity_check_matrix
        else:
            parity_check_matrix = self.parity_check_matrix

        n, k = self.n, self.k
        qe = len(qe_list)
        array = np.zeros((qe, 2*n), dtype=int)
        full_syndrome = np.zeros(s*(n-k), dtype=int)
        paulis = ['X', 'Y', 'Z']
        for dd in range(1, max_te+1):
            combos = combinations(product(range(qe), range(n)), dd)
            all_ops = product(paulis, repeat=dd)
            errors = product(all_ops, combos)
            for ops, inds in errors:
                for op, ind in zip(ops, inds):
                    if op == 'I':
                        continue
                    if op != 'X':
                        array[ind[0], ind[1]] = 1
                    if op != 'Z':
                        array[ind[0], n+ind[1]] = 1
                array = np.cumsum(array, axis=0)
                syndromes = array @ parity_check_matrix.T % 2
                for i, (q1, q2) in enumerate(zip(qe_list, qe_list[1:]+[s+1])):
                    full_syndrome[(q1-1)*(n-k):(q2-1)*(n-k)] = np.tile(syndromes[i], q2-q1)
                error_array = np.r_[array[-1][n:],array[-1][:n]]
                degen_int = self.get_degen_set(np.flip(syndromes[-1]), error_array)
                yield dd, full_syndrome, syndromes[-1], degen_int
                array[:] = 0


    def get_weight_statistics(self,
                          prob=None,
                          prob_e=None,
                          max_t=None,
                          max_te=None,
                          qe_list=[1],
                          fill_syndromes=False,
                          change_syndromes=True,
                          n_cx_errors=15,
                          truncate=True,
                          timeout=None,
                          memory_limit=None,
                          fake_meas=True,
                          show=False):
        """
        Calculate the failure rate of a quantum error correcting code.

        Parameters
        ----------
        prob : float or array_like, optional
            Probability of a Pauli error per qubit.
        prob_e : float or array_like, optional
            Probability of measurement error per qubit.
        max_t : int, optional
            Maximum number of time steps for the calculation.
        max_te : int, optional
            Maximum number of error time steps for the calculation.
        qe_list : list, optional
            List of qubits with measurement errors.
        fill_syndromes : bool, optional
            Fill the syndromes with errors, not used in this function.
        change_syndromes : bool, optional
            Change the syndromes using propagated syndrome generator.
        n_cx_errors : int, optional
            Maximum number of CNOT errors to consider.
        truncate : bool, optional
            Truncate the output array to the maximum time steps.
        timeout : float, optional
            Maximum time allowed for the calculation, not used in this function.
        memory_limit : float, optional
            Maximum memory allowed for the calculation, not used in this function.
        fake_meas : bool, optional
            Include fake measurements in the calculation.
        show : bool, optional
            Print the output if True, not used in this function.

        Returns
        -------
        failure_rate_all : ndarray
            Failure rate for all time steps and error probabilities.
        """

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        if fake_meas:
            self.n_meas = n_total - n - self._ignore_extra_stabs * (n - k)
            n_cx += self.n_meas

        max_t = n_cx if max_t is None else max_t
        n_show = max_t if truncate else n_cx

        # Choose appropriate syndrome conversion function based on problem size
        bin_array_to_int_syndrome = self.bin_array_to_int_fast if n_total - n <= 64 else self.bin_array_to_int
        bin_array_to_int_error = self.bin_array_to_int_fast if 2 * n <= 64 else self.bin_array_to_int

        # Convert input probabilities to numpy arrays
        prob = np.array([prob]) if isinstance(prob, float) else np.asarray(prob)
        prob_e = np.array([prob_e]) if isinstance(prob_e, float) else np.asarray(prob_e)

        syndrome_error_dict = {}
        nothing_array = np.zeros((max_te + 1, max_t + 1), dtype=int)
        syndrome_error_dict[0] = {(0, 0): nothing_array.copy()}
        syndrome_error_dict[0][(0, 0)][0, 0] = 1

        assert self.n_meas % (n - k) == 0
        s = self.n_meas // (n - k)

        # Iterate through syndrome/error pairs and populate syndrome_error_dict
        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(max_t, n_cx_errors)):
            hat_syndrome = syndrome[-(n - k):]
            main_syndrome = syndrome[:-(n - k)]
            int_hat_syndrome = bin_array_to_int_syndrome(hat_syndrome)
            int_main_syndrome = bin_array_to_int_syndrome(main_syndrome)
            degen_int = self.get_degen_set(hat_syndrome, error)

            # Update syndrome_error_dict with the new syndrome/error information
            if int_main_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_main_syndrome] = {(int_hat_syndrome, degen_int): nothing_array.copy()}
            elif (int_hat_syndrome, degen_int) not in syndrome_error_dict[int_main_syndrome]:
                syndrome_error_dict[int_main_syndrome][(int_hat_syndrome, degen_int)] = nothing_array.copy()

            syndrome_error_dict[int_main_syndrome][(int_hat_syndrome, degen_int)][0, dd] += 1

        # Update the syndromes based on the propagated errors
        for dd, error_main_syndrome, error_hat_syndrome, error_degen_int in self.propagated_syndrome_generator(change_syndromes, qe_list, max_te, s):
            for int_main_syndrome in list(syndrome_error_dict):
                main_syndrome = np.array(list(np.binary_repr(int_main_syndrome).zfill(self.n_meas))).astype(int)
                main_syndrome = (main_syndrome + error_main_syndrome) % 2
                new_int_main_syndrome = bin_array_to_int_syndrome(main_syndrome)

                for int_hat_syndrome, degen_int in list(syndrome_error_dict[int_main_syndrome]):
                    hat_syndrome = np.array(list(np.binary_repr(int_hat_syndrome).zfill(n - k))).astype(int)
                    hat_syndrome = (hat_syndrome + error_hat_syndrome) % 2
                    new_int_hat_syndrome = bin_array_to_int_syndrome(hat_syndrome)

                    degen_arr = np.array(list(np.binary_repr(degen_int).zfill(2 * k))).astype(int)
                    error_degen_arr = np.array(list(np.binary_repr(error_degen_int).zfill(2 * k))).astype(int)
                    degen_arr = (degen_arr + error_degen_arr) % 2
                    new_degen_int = bin_array_to_int_syndrome(degen_arr)

                    # Update syndrome_error_dict with the new error-propagated syndrome/error information
                    if new_int_main_syndrome not in syndrome_error_dict:
                        syndrome_error_dict[new_int_main_syndrome] = {(new_int_hat_syndrome, new_degen_int): nothing_array.copy()}
                    elif (new_int_hat_syndrome, new_degen_int) not in syndrome_error_dict[new_int_main_syndrome]:
                        syndrome_error_dict[new_int_main_syndrome][(new_int_hat_syndrome, new_degen_int)] = nothing_array.copy()

                    syndrome_error_dict[new_int_main_syndrome][(new_int_hat_syndrome, new_degen_int)][dd] += syndrome_error_dict[int_main_syndrome][(int_hat_syndrome, degen_int)][0]

        t_arr = np.arange(max_t + 1)[None, :, None, None]
        te_arr = np.arange(max_te + 1)[:, None, None, None]
        prob_e = prob_e[None, None, :, None]
        qe = len(qe_list)

        # Compute probabilities using syndrome_error_dict and input probabilities
        probabilities = np.zeros((max_te + 1, max_t + 1, prob_e.shape[2], prob.shape[0]))
        for _, degen_sets in syndrome_error_dict.items():
            degen_probs = np.zeros((len(degen_sets), max_te + 1, max_t + 1, prob_e.shape[2], prob.shape[0]))
            for i, (_, degen_set) in enumerate(degen_sets.items()):
                degen_probs[i] = degen_set[:, :, None, None] * (prob / n_cx_errors) ** t_arr * (1 - prob) ** (n_cx - t_arr) * (prob_e / 3) ** te_arr * (1 - prob_e) ** (qe * n - te_arr)
            degen_probs = np.cumsum(degen_probs, axis=1)
            degen_probs = np.cumsum(degen_probs, axis=2)
            optimal_prob = np.amax(degen_probs, axis=0)
            probabilities += optimal_prob

        failure_rate_all = 1 - probabilities

        return failure_rate_all

    def get_weight_statistics_perfect(self,
                                    prob=None,
                                    max_t=None,
                                    fill_syndromes=False,
                                    change_syndromes=True,
                                    qe_list=[1],
                                    n_cx_errors=15,
                                    prep_errors=True,
                                    truncate=True,
                                    timeout=None,
                                    memory_limit=None,
                                    fake_meas=True,
                                    show=False):

        # warnings.warn("""This function is a lower bound for the performance 
        #                 of the code. It assumes that errors are distinct. 
        #                 Therefore, it not only disregards possible degeneracies
        #                 in the main n qubits, but also the highly degenerate 
        #                 errors in the ancilla qubits.""")
        start_time = time()
        
        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        if fake_meas:
            self.n_meas = n_total-n - self._ignore_extra_stabs*(n-k)
            n_cx += self.n_meas * (2 if prep_errors else 1)

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

        if 2*n <= 64:
            bin_array_to_int_error = self.bin_array_to_int_fast
        else:
            bin_array_to_int_error = self.bin_array_to_int

        # Converting prob to numpy array
        if isinstance(prob, float):
            prob = [prob]

        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        syndrome_error_dict = {}
        zero_array = np.zeros(max_t+1, dtype=int)
        zero_array[0] = 1
        one_array = np.zeros(max_t+1, dtype=int)
        one_array[1] = 1
        nothing_array = np.zeros(max_t, dtype=int)
        #nothing_array = np.zeros((max_te+1, max_t+1), dtype=int)
        syndrome_error_dict[0] = {0:nothing_array.copy()}
        #syndrome_error_dict[0][(0,0)][0,0] = 1

        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        #probability_no_error = np.array([(1-p)**n_cx for p in prob])
        #fail_size = int(sum(n_cx_errors**i * comb(n_cx, i) for i in range(1, max_t+1)))
        #box_size = 10_000
        #fail_arr = np.zeros(fail_size//box_size + 1)
        #fail = 0
        #loops = np.zeros(3, dtype=int)
        memory_size = 0
        assert self.n_meas % (n-k) == 0
        s = self.n_meas // (n-k)
        for i, (dd, _, hat_syndrome, degen_int) in enumerate(self.propagated_syndrome_generator(change_syndromes, qe_list, max_t, s)):
            #print(syndrome)
            #hat_syndrome = syndrome[-(n-k):]
            #main_syndrome = syndrome[:-(n-k)]
            int_hat_syndrome = bin_array_to_int_syndrome(hat_syndrome)
            #int_main_syndrome = bin_array_to_int_syndrome(main_syndrome)
            #degen_int = self.get_degen_set(hat_syndrome, error)

            if int_hat_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_hat_syndrome] = {degen_int:nothing_array.copy()}
                #syndrome_error_dict[int_main_syndrome][(int_hat_syndrome,degen_int)][dd-1] += 1
                memory_size += 1
                #case = 0

            elif degen_int in syndrome_error_dict[int_hat_syndrome]:
                #syndrome_error_dict[int_main_syndrome][(int_hat_syndrome,degen_int)][dd-1] += 1
                #case = 1
                pass

            else:
                syndrome_error_dict[int_hat_syndrome][degen_int] = nothing_array.copy()
                #case = 2
                memory_size += 1
            syndrome_error_dict[int_hat_syndrome][degen_int][dd-1] += 1

            #loops[case] += 1
            if i%10000==0:
                if timeout is not None:
                    end_time = time()
                    if end_time - start_time > timeout:
                        print("Timeout!")
                        broke_out = True
                        break
                if memory_limit is not None:
                    if memory_size > memory_limit:
                        print("Out of memory!")
                        broke_out = True
                        break

        #self.loops = loops

        t_arr = 1 + np.arange(max_t)[:, None]
        t_arr_0 = np.arange(max_t+1)[:, None]

        probabilities = np.zeros((max_t, prob.shape[0]))
        for int_syndrome, degen_sets in syndrome_error_dict.items():
            if int_syndrome==0:
                degen_probs = np.zeros((len(degen_sets), max_t+1, prob.shape[0]))
                for i, (degen_tup, degen_set) in enumerate(degen_sets.items()):
                    if degen_tup == 0:
                        degen_set = np.insert(degen_set, 0, 1)
                    else:
                        degen_set = np.insert(degen_set, 0, 0)
                    degen_probs[i] = degen_set[:, None] * (prob/3)**t_arr_0 * (1-prob)**(n-t_arr_0)
                #print(int_syndrome, degen_probs)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)[1:]
                probabilities += optimal_prob
            else:
                degen_probs = np.zeros((len(degen_sets), max_t, prob.shape[0]))
                for i, (_, degen_set) in enumerate(degen_sets.items()):
                    degen_probs[i] = degen_set[:, None] * (prob/3)**t_arr * (1-prob)**(n-t_arr)
                #print(int_syndrome, degen_probs)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)
                probabilities += optimal_prob

            failure_rate_all = 1 - probabilities

        #self.fail_arr = fail_arr
        
        #self.syndrome_count = len(syndrome_error_dict)
        #self.syndrome_error_dict = syndrome_error_dict
        
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

        start_time = time()

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total
        if fake_meas:
            self.n_meas = n_total-n - self._ignore_extra_stabs*(n-k)
            n_cx += self.n_meas
            
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

            hat_syndrome = syndrome[-(n-k):]
            main_syndrome = syndrome[:-(n-k)]
            int_hat_syndrome = bin_array_to_int_syndrome(hat_syndrome)
            int_main_syndrome = bin_array_to_int_syndrome(main_syndrome)
            degen_int = self.get_degen_set(hat_syndrome, error)
            degen_tup = (int_hat_syndrome,degen_int) 

            if int_main_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_main_syndrome] = {degen_tup:nothing_array.copy()}

            elif degen_tup not in syndrome_error_dict[int_main_syndrome]:
                syndrome_error_dict[int_main_syndrome][degen_tup] = nothing_array.copy()
            
            syndrome_error_dict[int_main_syndrome][degen_tup][dd-1] += 1

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
                            int_degen = (int_degen_1[0] ^ int_degen_big[0],
                                            int_degen_1[1] ^ int_degen_big[1])

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
                        syndrome_error_dict[int_syndrome][degen_int][dd-1] = a // dd
            else:
                for int_syndrome in list(syndrome_error_dict):
                    degen_sets = syndrome_error_dict[int_syndrome]
                    for degen_int, degen_set in degen_sets.items():
                        if int_syndrome==0 and degen_int==(0,0):
                            a = degen_set[dd-1] - n_cx_errors*(n_cx - (dd-2)) - (n_cx_errors-1)*(dd-1)*degen_set[dd-2]
                            syndrome_error_dict[int_syndrome][degen_int][dd-1] = a // dd
                        else:
                            a = degen_set[dd-1] - (n_cx_errors-1)*(dd-1)*degen_set[dd-2]
                            syndrome_error_dict[int_syndrome][degen_int][dd-1] = a // dd

            print(dd, len(syndrome_error_dict), len_dict_1*len(syndrome_error_dict))

            if broke_out:
                break

        t_arr = 1 + np.arange(max_t)[:, None]
        t_arr_0 = np.arange(max_t+1)[:, None]

        probabilities = np.zeros((max_t, prob.shape[0]))
        for int_syndrome, degen_sets in syndrome_error_dict.items():
            if int_syndrome==0:
                degen_probs = np.zeros((len(degen_sets), max_t+1, prob.shape[0]))
                for i, (degen_int, degen_set) in enumerate(degen_sets.items()):
                    if degen_int == (0,0):
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
            
        return failure_rate_all
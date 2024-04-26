from .MeasFaultTolerance import MeasFaultTolerance
import numpy as np
from scipy.special import comb
from copy import deepcopy
from time import time

class RealisticSimulator(MeasFaultTolerance):

    def get_weight_statistics(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            prep_errors=True,
                            truncate=True,
                            only_distinct=False,
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

        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])
        #fail_size = int(sum(n_cx_errors**i * comb(n_cx, i) for i in range(1, max_t+1)))
        #box_size = 10_000
        #fail_arr = np.zeros(fail_size//box_size + 1)
        #fail = 0
        #loops = np.zeros(3, dtype=int)
        memory_size = 0
        syndrome_error_dict[0] = {(0,0):nothing_array.copy()}
        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(max_t, n_cx_errors, prep_errors)):
            #print(syndrome)
            hat_syndrome = syndrome[-(n-k):]
            main_syndrome = syndrome[:-(n-k)]
            int_hat_syndrome = bin_array_to_int_syndrome(hat_syndrome)
            int_main_syndrome = bin_array_to_int_syndrome(main_syndrome)
            degen_int = self.get_degen_set(hat_syndrome, error)

            if int_main_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_main_syndrome] = {(int_hat_syndrome,degen_int):nothing_array.copy()}
                syndrome_error_dict[int_main_syndrome][(int_hat_syndrome,degen_int)][dd-1] += 1
                memory_size += 1
                #case = 0

            elif (int_hat_syndrome,degen_int) in syndrome_error_dict[int_main_syndrome]:
                syndrome_error_dict[int_main_syndrome][(int_hat_syndrome,degen_int)][dd-1] += 1
                #case = 1

            else:
                if only_distinct:
                    pass
                else:
                    syndrome_error_dict[int_main_syndrome][(int_hat_syndrome,degen_int)] = nothing_array.copy()
                    syndrome_error_dict[int_main_syndrome][(int_hat_syndrome,degen_int)][dd-1] += 1
                    #case = 2
                    memory_size += 1

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
                    if degen_tup == (0,0):
                        degen_set = np.insert(degen_set, 0, 1)
                    else:
                        degen_set = np.insert(degen_set, 0, 0)
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr_0 * (1-prob)**(n_cx-t_arr_0)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)[1:]
                probabilities += optimal_prob
            else:
                degen_probs = np.zeros((len(degen_sets), max_t, prob.shape[0]))
                for i, (_, degen_set) in enumerate(degen_sets.items()):
                    degen_probs[i] = degen_set[:, None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)
                degen_probs = np.cumsum(degen_probs, axis=1)
                optimal_prob = np.amax(degen_probs, axis=0)
                probabilities += optimal_prob

            failure_rate_all = 1 - probabilities

        #self.fail_arr = fail_arr
        self.syndrome_table = syndrome_error_dict
        self.syndrome_count = len(syndrome_error_dict)
        
        return failure_rate_all
    
    def get_weight_statistics_fast(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            prep_errors=True,
                            truncate=True,
                            timeout=None,
                            memory_limit=None,
                            fake_meas=True,
                            show=False):

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

        # Converting prob to numpy array
        if isinstance(prob, float):
            prob = [prob]

        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        syndrome_error_dict = {}
        nothing_array = np.zeros(max_t, dtype=int)

        probabilities = np.zeros((n_show, prob.shape[0]))
        memory_size = 0

        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(1, n_cx_errors, prep_errors)):

            #int_syndrome = bin_array_to_int_syndrome(syndrome)
            #degen_int = self.get_degen_set(syndrome, error)
            hat_syndrome = syndrome[-(n-k):]
            main_syndrome = syndrome[:-(n-k)]
            int_hat_syndrome = bin_array_to_int_syndrome(hat_syndrome)
            int_main_syndrome = bin_array_to_int_syndrome(main_syndrome)
            degen_int = self.get_degen_set(hat_syndrome, error)
            degen_tup = (int_hat_syndrome,degen_int) 

            if int_main_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_main_syndrome] = {degen_tup:nothing_array.copy()}
                memory_size += 1

            elif degen_tup not in syndrome_error_dict[int_main_syndrome]:
                syndrome_error_dict[int_main_syndrome][degen_tup] = nothing_array.copy()
                memory_size += 1
            
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
                                memory_size += 1

                            elif int_degen not in syndrome_error_dict[int_syndrome]:
                                syndrome_error_dict[int_syndrome][int_degen] = nothing_array.copy()
                                memory_size += 1
                            
                            syndrome_error_dict[int_syndrome][int_degen][dd-1] += data_1[0] * data_2[dd-2]

        
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

        #self.syndrome_error_dict = dict(sorted(syndrome_error_dict.items()))

        return failure_rate_all
from .FaultTolerance import FaultTolerance
from .GoodCode import GoodCode
import numpy as np
from scipy.special import comb
from itertools import combinations, product
from time import time

class DegenerateFaultTolerance(FaultTolerance):

    def get_base_errors(self):

        self.base_errors = np.zeros((self.n_cx, len(self.gate_error_list), 2*self.n), dtype=int)
        for i, _ in enumerate(self.base_errors):
            for j, _ in enumerate(self.base_errors[0]):
                error = self.ev_error_list[i][j]
                if error[0] == '-':
                    error = error[1:]
                error = error[:self.n]

                # Convert error to int array
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

        n, k = self.n, self.k
        n_cx = self.n_cx
        n_total = self.n_total

        array = np.zeros(n_total-n, dtype=int)
        err_array = np.zeros(2*n, dtype=int)
        # paulis = ['X', 'Y', 'Z']
        # syndrome_time, error_time = 0., 0.
        # loops = 0
        for dd in range(1, max_t+1):
            combos = combinations(range(n_cx), dd)
            all_ops = product(range(1, n_errors+1), repeat=dd)
            case = product(all_ops, combos)
            for error_nums, inds in case:
                for ind_err, ind_cx in zip(error_nums, inds):
                    #print(ind_err, ind_cx)
                    #start_time = time()
                    syndrome = self.syndrome_power_set(ind_err, ind_cx)
                    #middle_time = time()
                    array = (array + syndrome) % 2
                    error = self.error_power_set(ind_err, ind_cx)
                    #end_time = time()
                    err_array = (err_array + error) % 2
                    #syndrome_time += middle_time-start_time
                    #error_time += end_time-middle_time
                    #loops += 1
                yield dd, array, err_array
                array[:] = 0
                err_array[:] = 0

    def error_power_set(self, ind_err, ind_cx):
        errors = self.base_errors[ind_cx]
        array = np.zeros(2*self.n, dtype=int)
        #bin_rep = np.array(list(np.binary_repr(ind_err).zfill(len(self.gate_error_list)))).astype(int)
        for i, ind in enumerate(self.bin_rep[ind_err]):
            if ind==1:
                array = (array + errors[i]) % 2
        return array

    def set_parity_check_matrix(self, row_echelon=True):

        parity_check_matrix = np.zeros((self.n-self.k, 2*self.n), dtype=int)
        for i, stab in enumerate(self.stabilizers):
            stab = list(stab)[1:]
            array = np.zeros(self.n*2, dtype=int)
            for ind, op in enumerate(reversed(stab)):
                        if op == 'I':
                            continue
                        if op != 'X':
                            array[self.n+ind] = 1
                        if op != 'Z':
                            array[ind] = 1
            parity_check_matrix[i] = array
        self.parity_check_matrix = parity_check_matrix

        if row_echelon:
            M = parity_check_matrix.copy()

            lead = 0
            n_rows, n_cols = M.shape
            for r in range(n_rows):
                #print(M)
                if n_cols <= lead:
                    break
                i = r
                while M[i, lead] == 0:
                    i += 1
                    if n_rows == i:
                        i = r
                        lead += 1
                        if n_cols == lead:
                            break
                if i != r:
                    M[[i,r]] = M[[r,i]]
                #Divide row r by M[r, lead]
                rows_to_change = np.nonzero(M[:, lead])[0]
                rows_to_change = rows_to_change[rows_to_change!=r]
                #print(lead, rows_to_change)
                M[rows_to_change, lead:] = (M[rows_to_change, lead:] + M[r, lead:]) % 2
                lead += 1
            
            self.rre_parity_check_matrix = M

            pivots = np.zeros(M.shape[1], dtype=int) - 1
            #print(M.shape, pivots.shape)
            shift = 0
            for i in range(M.shape[0]):
                pivot_is_found = False
                while not pivot_is_found:
                    if M[i, i+shift] == 1:
                        pivot_is_found = True
                        pivots[i+shift] = i
                    else:
                        shift += 1
            self.pivots = pivots


    def get_weight_statistics_old(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            truncate=True,
                            consider_degeneracy=False,
                            show=False):

        # warnings.warn("""This function is a lower bound for the performance 
        #                 of the code. It assumes that errors are distinct. 
        #                 Therefore, it not only disregards possible degeneracies
        #                 in the main n qubits, but also the highly degenerate 
        #                 errors in the ancilla qubits.""")

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

        if max_t is None:
            max_t = n_cx

        if truncate:
            n_show = max_t
        else:
            n_show = n_cx

        # Converting prob to numpy array
        if isinstance(prob, float):
            prob = [prob]

        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        # if fill_syndromes:
        #     syndrome_list = np.zeros(2**(n_total-n), dtype=bool)
        #     syndrome_list[0] = True
        # else:
        #seen_syndromes = set([0])
        syndrome_error_dict = {0:0}
        # zero_array = np.zeros(max_t+1, dtype=int)
        # zero_array[0] = 1
        # one_array = np.zeros(max_t+1, dtype=int)
        # one_array[1] = 1
        # two_array = np.zeros(max_t+1, dtype=int)
        # two_array[2] = 1
        # unit_array = np.zeros((3, max_t+1), dtype=int)
        # unit_array[0] = zero_array
        # unit_array[1] = one_array
        # unit_array[2] = two_array
        # prob_data = {0:zero_array}

        accounted_syndromes = 1
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))
        if consider_degeneracy:
            error_type_degen = np.zeros(n_show, dtype=int)
            probabilities_degen = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])

        # dict_case = {1:'N',2:'R',3:'D',4:'F'}

        # end_time = time()
        # loops = {1:0,2:0,3:0,4:0}
        # syndrome_time = 0.
        # analysis_time = {1:0.,2:0.,3:0.,4:0.}
        fail_size = int(sum(n_cx_errors**i * comb(n_cx, i) for i in range(1, max_t+1)))
        fail_arr = np.zeros(fail_size)
        fail = 0
        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(max_t, n_cx_errors)):

            # out = 0
            # for bit in syndrome:
            #     out = (out << 1) | bool(bit)
            # int_syndrome = out

            # out = 0
            # for bit in error:
            #     out = (out << 1) | bool(bit)
            # int_error = out
            #start_time = time()

            int_syndrome = self.bin_array_to_int(syndrome)
            int_error = self.bin_array_to_int(error)

            if int_syndrome not in syndrome_error_dict:
                #seen_syndromes.add(int_syndrome)
                syndrome_error_dict[int_syndrome] = int_error
                #prob_data[int_syndrome] = unit_array[dd].copy()
                #accounted_syndromes += 1
                error_type[dd-1] += 1
                #case = 1
                # if accounted_syndromes == 2**(n_total-n):
                #     break
            
            # Test if error appeared before
            elif int_error == syndrome_error_dict[int_syndrome]:
                error_type[dd-1] += 1
                #prob_data[int_syndrome] += unit_array[dd]
                #case = 2

            elif consider_degeneracy and self.error_is_degenerate(int_error, syndrome_error_dict[int_syndrome]):
                error_type_degen[dd-1] += 1
                #prob_data[int_syndrome] += unit_array[dd]
                #case = 3

            else:
                fail += 1
                #case = 4

            if i%100000==0:
                print("Fail {}: {}".format(i//100000, fail/100000))
                fail_arr[i//100000] = fail/100000
                fail = 0

            #print(dict_case[case], dd, syndrome, error) if show else None

        #     loops[case] += 1
        #     syndrome_time += start_time - end_time
        #     end_time = time()
        #     analysis_time[case] += end_time - start_time

        # total_loops = loops[1] + loops[2] + loops[3] + loops[4]
        # print(total_loops, syndrome_time/total_loops)
        # for case in range(1, 5):
        #     print(dict_case[case], loops[case], analysis_time[case]/(loops[case] if loops[case]!=0 else 1.))

        t_arr = 1 + np.arange(max_t)[:, None]
        probabilities = error_type[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)
        #print(error_type, probabilities)
        if consider_degeneracy:
            probabilities_degen = error_type_degen[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)

        failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

        n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
        correctable_fraction = error_type / n_errors
        print("Errors:", n_errors)
        print(fail_arr)

        mean_iterations = np.array([
            sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                    sum(error_type[:dd]) + (nn+1)/2
                ) for dd, nn in enumerate(error_type[:i])
            ) for i in range(1,n_show+1)
        ])

        if consider_degeneracy:
            error_type += error_type_degen
            probabilities += probabilities_degen

            # prob_data = dict(sorted(prob_data.items()))
            # print(np.array(list(prob_data.values())))
            # print(error_type)

            failure_rate_all = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

            n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
            correctable_fraction_all = error_type / n_errors

            mean_iterations_all = np.array([
                sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                        sum(error_type[:dd]) + (nn+1)/2
                    ) for dd, nn in enumerate(error_type[:i])
                ) for i in range(1,n_show+1)
            ])

        if not consider_degeneracy:
            return failure_rate, correctable_fraction, mean_iterations
        else:
            return (failure_rate, failure_rate_all), (correctable_fraction, correctable_fraction_all), (mean_iterations, mean_iterations_all)

    def get_weight_statistics(self,
                            prob=None,
                            max_t=None,
                            fill_syndromes=False,
                            n_cx_errors=15,
                            truncate=True,
                            consider_degeneracy=False,
                            show=False):

        # warnings.warn("""This function is a lower bound for the performance 
        #                 of the code. It assumes that errors are distinct. 
        #                 Therefore, it not only disregards possible degeneracies
        #                 in the main n qubits, but also the highly degenerate 
        #                 errors in the ancilla qubits.""")

        n, k, n_cx, n_total = self.n, self.k, self.n_cx, self.n_total

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

        syndrome_error_dict = {0:0}

        accounted_syndromes = 1
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))
        if consider_degeneracy:
            error_type_degen = np.zeros(n_show, dtype=int)
            probabilities_degen = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])
        fail_size = int(sum(n_cx_errors**i * comb(n_cx, i) for i in range(1, max_t+1)))
        box_size = 10_000
        fail_arr = np.zeros(fail_size//box_size + 1)
        fail = 0
        for i, (dd, syndrome, error) in enumerate(self.syndrome_generator(max_t, n_cx_errors)):

            int_syndrome = bin_array_to_int_syndrome(syndrome)
            int_error = bin_array_to_int_error(error)

            if int_syndrome not in syndrome_error_dict:
                syndrome_error_dict[int_syndrome] = int_error
                error_type[dd-1] += 1
            
            # Test if error appeared before
            elif int_error == syndrome_error_dict[int_syndrome]:
                error_type[dd-1] += 1

            elif consider_degeneracy and self.error_is_degenerate(int_error, syndrome_error_dict[int_syndrome]):
                error_type_degen[dd-1] += 1

            else:
                fail += 1

            if i%box_size==0:
                print("Fail {}: {}".format(i//box_size, fail/box_size))
                fail_arr[i//box_size] = fail/box_size
                fail = 0

        t_arr = 1 + np.arange(max_t)[:, None]
        probabilities = error_type[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)
        #print(error_type, probabilities)
        if consider_degeneracy:
            probabilities_degen = error_type_degen[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)

        failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

        n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
        correctable_fraction = error_type / n_errors
        self.fail_arr = fail_arr

        mean_iterations = np.array([
            sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                    sum(error_type[:dd]) + (nn+1)/2
                ) for dd, nn in enumerate(error_type[:i])
            ) for i in range(1,n_show+1)
        ])

        if consider_degeneracy:
            error_type += error_type_degen
            probabilities += probabilities_degen

            # prob_data = dict(sorted(prob_data.items()))
            # print(np.array(list(prob_data.values())))
            # print(error_type)

            failure_rate_all = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

            n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
            correctable_fraction_all = error_type / n_errors

            mean_iterations_all = np.array([
                sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                        sum(error_type[:dd]) + (nn+1)/2
                    ) for dd, nn in enumerate(error_type[:i])
                ) for i in range(1,n_show+1)
            ])

        if not consider_degeneracy:
            return failure_rate, correctable_fraction, mean_iterations
        else:
            return (failure_rate, failure_rate_all), (correctable_fraction, correctable_fraction_all), (mean_iterations, mean_iterations_all)

    def error_is_degenerate(self, int_error_1, int_error_2):
        n = self.n
        error_1 = np.array(list(np.binary_repr(int_error_1).zfill(n*2))).astype(int)
        error_2 = np.array(list(np.binary_repr(int_error_2).zfill(n*2))).astype(int)
        error = (error_1 + error_2) % 2
        # result_1 = GoodCode._get_boolean_solution(self.parity_check_matrix.T,
        #                                         np.r_[error[n:],error[:n]])
        result = GoodCode._get_boolean_solution_3(self.rre_parity_check_matrix,
                                                self.pivots,
                                                error)
        # if result_1 != result_2:
        #     print(result_1, result_2, "Different!")
        return result


    @staticmethod
    def bin_array_to_int(array):

        out = 0
        for bit in array:
            out = (out << 1) | bool(bit)
        int_array = out
        
        return int_array

    @staticmethod
    def bin_array_to_int_fast(array):

        out = 0
        for bit in array.astype(bool):
            out = (out << 1) | bit
        int_array = out
        
        return int_array

    def get_weight_statistics_fast(self,
                            prob=None,
                            max_t=None,
                            n_cx_errors=15,
                            truncate=True,
                            show=False):

        # warnings.warn("""This function is a lower bound for the performance 
        #                 of the code. It assumes that errors are distinct. 
        #                 Therefore, it not only disregards possible degeneracies
        #                 in the main n qubits, but also the highly degenerate 
        #                 errors in the ancilla qubits.""")

        n, n_cx, n_total = self.n, self.n_cx, self.n_total

        if max_t is None:
            max_t = n_cx

        if truncate:
            n_show = max_t
        else:
            n_show = n_cx

        # Converting prob to numpy array
        if isinstance(prob, float):
            prob = [prob]

        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)

        # if fill_syndromes:
        #     syndrome_list = np.zeros(2**(n_total-n), dtype=bool)
        #     syndrome_list[0] = True
        # else:
        #seen_syndromes = set([0])
        syndrome_error_dict = {0:0}
        zero_array = np.zeros(max_t+1, dtype=int)
        zero_array[0] = 1
        one_array = np.zeros(max_t+1, dtype=int)
        one_array[1] = 1
        prob_data = {0:zero_array}
        #error_data = {0:}
        #max_size = 2**(n_total-n)

        #accounted_syndromes = 1
        error_type = np.zeros(n_show, dtype=int)
        probabilities = np.zeros((n_show, prob.shape[0]))

        probability_no_error = np.array([(1-p)**n_cx for p in prob])

        dict_case = {1:'N',2:'R',3:'D',4:'F'}

        end_time = time()
        loops = {1:0,2:0,3:0,4:0}
        syndrome_time = 0.
        analysis_time = {1:0.,2:0.,3:0.,4:0.}
        for i,(dd, syndrome, error) in enumerate(self.syndrome_generator(1, n_cx_errors)):

            start_time = time()

            int_syndrome = self.bin_array_to_int(syndrome)
            int_error = self.bin_array_to_int(error)

            if int_syndrome not in syndrome_error_dict:
                #seen_syndromes.add(int_syndrome)
                syndrome_error_dict[int_syndrome] = int_error
                prob_data[int_syndrome] = one_array.copy()
                #accounted_syndromes += 1
                #error_type[dd-1] += 1
                case = 1
                # if len(syndrome_error_dict) == max_size:
                #     break
            
            # Test if error appeared before
            elif int_error == syndrome_error_dict[int_syndrome]:
                #error_type[dd-1] += 1
                prob_data[int_syndrome] += one_array
                case = 2

            elif self.error_is_degenerate(int_error, syndrome_error_dict[int_syndrome]):
                #error_type[dd-1] += 1
                prob_data[int_syndrome] += one_array
                case = 3

            else:
                case = 4

            print(dict_case[case], dd, syndrome, error) if show else None

            loops[case] += 1
            syndrome_time += start_time - end_time
            end_time = time()
            analysis_time[case] += end_time - start_time

        #prob_data_1 = prob_data.copy()
        prob_data_1 = dict(sorted(prob_data.items()))
        #syndrome_error_dict_1 = syndrome_error_dict.copy()
        #print(prob_data_1)
        print(np.array(list(prob_data_1.values())))
        error_type = np.sum(np.array(list(prob_data_1.values())), axis=0)[1:]
        print(error_type)
        error_len_1 = len(prob_data)

        for dd in range(2, max_t+1):
            prob_data_old = prob_data.copy()
            syndrome_error_dict_old = syndrome_error_dict.copy()
            prob_data = {}
            syndrome_error_dict = {}
            for i,(int_syndrome_2, data_2) in enumerate(prob_data_old.items()):
                for j,(int_syndrome_1, data_1) in enumerate(prob_data_1.items()):

                    print(i,j)

                    start_time = time()

                    int_syndrome = int_syndrome_1 ^ int_syndrome_2
                    int_error = syndrome_error_dict_old[int_syndrome_1] ^ syndrome_error_dict_old[int_syndrome_2]

                    if int_syndrome not in syndrome_error_dict:
                        #seen_syndromes.add(int_syndrome)
                        syndrome_error_dict[int_syndrome] = int_error
                        prob_data[int_syndrome] = self.join_probs(data_1, data_2)
                        #accounted_syndromes += 1
                        #error_type[dd-1] += 1
                        case = 1
                        # if len(syndrome_error_dict) == max_size:
                        #     break
                    
                    # Test if error appeared before
                    elif int_error == syndrome_error_dict[int_syndrome]:
                        #error_type[dd-1] += 1
                        prob_data[int_syndrome] += self.join_probs(data_1, data_2)
                        case = 2

                    elif self.error_is_degenerate(int_error, syndrome_error_dict[int_syndrome]):
                        #error_type[dd-1] += 1
                        prob_data[int_syndrome] += self.join_probs(data_1, data_2)
                        case = 3

                    else:
                        case = 4

                    syndrome = np.array(list(np.binary_repr(int_syndrome).zfill(n-self.k))).astype(int)
                    error = np.array(list(np.binary_repr(int_error).zfill(n*2))).astype(int)
                    print(dict_case[case], dd, syndrome, error) if show else None

                    loops[case] += 1
                    syndrome_time += start_time - end_time
                    end_time = time()
                    analysis_time[case] += end_time - start_time

        total_loops = loops[1] + loops[2] + loops[3] + loops[4]
        print(total_loops, syndrome_time/total_loops)
        for case in range(1, 5):
            print(dict_case[case], loops[case], analysis_time[case]/(loops[case] if loops[case]!=0 else 1.))

        #print(prob_data)
        #prob_data[0][1] = prob_data[0][1] // 2
        prob_data = dict(sorted(prob_data.items()))
        prob_data[0][2] -= error_len_1-1
        for key in prob_data:
            # if key==0:
            #     pass
            # else:
                # if prob_data[key][1]%2==1 or prob_data[key][2]%2==1:
                #     print(key, prob_data[key])
            prob_data[key] = prob_data[key] // 2
        print(np.array(list(prob_data.values())))
        error_type = np.sum(np.array(list(prob_data.values())), axis=0)[1:]
        print(error_type)

        t_arr = 1 + np.arange(max_t)[:, None]
        probabilities = error_type[:,None] * (prob/n_cx_errors)**t_arr * (1-prob)**(n_cx-t_arr)

        failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

        n_errors = np.array([n_cx_errors**i * comb(n_cx, i) for i in range(1, n_show+1)])
        correctable_fraction = error_type / n_errors

        mean_iterations = np.array([
            sum((prob/n_cx_errors)**(dd+1) * (1-prob)**(n_cx-dd-1) * nn * (
                    sum(error_type[:dd]) + (nn+1)/2
                ) for dd, nn in enumerate(error_type[:i])
            ) for i in range(1,n_show+1)
        ])
        
        return failure_rate, correctable_fraction, mean_iterations

    @staticmethod
    def join_probs(data_1, data_all):
        data = np.roll(data_1[1] * data_all, 1)
        if data_1[0] != 0:
            data += data_1[0] * data_all
        return data
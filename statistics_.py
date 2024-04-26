import numpy as np
from scipy.special import comb
from scipy.stats import binom
from .pauli import pauli_generator_num, pauli_generator
import qiskit.quantum_info as qi

def get_uniform_at_random_statistics(n,
                                    x,
                                    p,
                                    max_t):
    
    any_error_prob = 1 - (1-p)**n

    #E_N = np.array([(comb(n, i+1)) * (3**(i+1) / 2**(n*(1-x))) for i in range(max_t)])
    E_N = np.array([(comb(n, i+1)) * np.exp((i+1)*np.log(3) - (n*(1-x))*np.log(2)) for i in range(max_t)])
    #B_N = np.insert(np.cumsum(E_N, axis=0)[:-1], 0, 1/ 2**(n*(1-x)), 0)
    B_N = np.insert(np.cumsum(E_N, axis=0)[:-1], 0, np.exp(-n*(1-x)*np.log(2)), 0)
    y_theory = np.where(E_N < 1e-3, 1, (1-np.exp(-E_N))/E_N) * np.exp(-B_N)

    dd = np.arange(1, max_t+1)[:,None, None]
    #failure_theory = any_error_prob - np.cumsum(p**dd * (1-p)**(n-dd) * comb(n, dd) * y_theory[:,:,None], axis=0)
    failure_theory = any_error_prob - np.cumsum(binom.pmf(dd, n, p) * y_theory[:,:,None], axis=0)

    return y_theory, failure_theory, any_error_prob

def get_weight_statistics(n,
                        k,
                        parity_check_matrix,
                        prob=None,
                        max_t=None,
                        fill_syndromes=False,
                        errors='all'):
    """Statistics for a code, based on the error weight.

    Bernoulli noise is assumed, where the probability of an error occuring in 
    any qubit is given in `prob`. 
    
    Parameters
    ----------
    n : int
        Number of qubits in the encoding.
    k : int
        Original number of data qubits.
    parity_check_matrix : 2D array
        ((n-k) x 2n) 2D binary array, representing the parity check matrix of 
        the code.
    prob : float, list, 1D array
        Probabilities to consider, independently, for the statistics.
    max_t : int
        Maximum weight considered for the analysis. Defaults to `n`, although 
        that is too computationally hard for high `n`.
    fill_syndromes : bool, default True
        There are two approaches to check if an error has a syndrome already 
        associated with a more likely error. If ``True``, it creates a 
        syndrome table containing all possible syndromes and keeps track of 
        which syndromes have been seen before. If ``False``, adds any newly 
        seen syndrome to a set containing previously seen syndromes, and uses 
        that set to check if syndromes have been seen before. If the syndrome 
        table is small, the default option might be faster, but for large 
        (n-k), the syndrome table would be too large to store efficiently in 
        memory.
    errors : {'all', 'X'}, default 'all'
        Type of error statistics. If ``'all'``, it is depolarizing noise, and 
        it assumes that the X, Y, and Z errors are all equally likely, with 
        probability equal to 1/3. If ``'X'``, then only one error (such as X) 
        is assumed to occur.

    Returns
    -------
    failure_rate : 2D array
        (n x prob.size) 2D array, containing the probability of the code 
        successfully correcting an error of weight `t` (with `t` going from 1 
        to `n` in the first axis), for each probability in `prob` (second 
        axis). Note that, if `max_t` < `n`, the latter rows are redundant. 
        They are only kept there to keep the output in a consistent size, for 
        varying `max_t`.
    correctable_fraction : 1D array
        n-sized array containing the fraction of errors of weight `t` that can
        be corrected (assuming that lower weight errors take precedence over 
        higher weight ones). The latter entries may be redundant.
    mean_iterations : 2D array
        (n x prob.size) 2D array, containing the mean number of iterations 
        that GRAND, with a membership test, would need to go through to 
        correct the errors it CAN correct.

    Warnings
    --------
    The option `errors` = ``'X'`` as never been properly tested. It is not 
    clear which errors the `mean_iterations` actually accounts for.

    This function does not consider degenerate scenarios.

    Notes
    -----
    This function considers all possibilities, and does not use Monte Carlo 
    approximations, so its results are exact, for the given parameters.
    """
    

    # Converting prob to numpy array
    if isinstance(prob, float):
        prob = [prob]

    if not isinstance(prob, np.ndarray):
        prob = np.array(prob)

    if fill_syndromes:
        syndrome_list = np.zeros(2**(n-k), dtype=bool)
        syndrome_list[0] = True
    else:
        seen_syndromes = set([0])

    accounted_syndromes = 1
    error_type = np.zeros(n, dtype=int)
    probabilities = np.zeros((n, prob.shape[0]))

    if max_t is None:
        max_t = n

    probability_no_error = np.array([(1-p)**n for p in prob])

    for dd, error in pauli_generator_num(n, max_t+1):

        syndrome = error @ parity_check_matrix.T % 2

        out = 0
        for bit in syndrome:
            out = (out << 1) | bit
        int_syndrome = out

        if fill_syndromes:
            if not syndrome_list[int_syndrome]:
                syndrome_list[int_syndrome] = True
                accounted_syndromes += 1

                error_type[dd-1] += 1

                if errors == 'all':
                    probabilities[dd-1] += (prob/3)**dd * (1-prob)**(n-dd)
                elif errors == 'X':
                    probabilities[dd-1] += prob**dd * (1-prob)**(n-dd)

                if accounted_syndromes == 2**(n-k):
                    break
        else:
            if int_syndrome not in seen_syndromes:
                seen_syndromes.add(int_syndrome)
                accounted_syndromes += 1

                error_type[dd-1] += 1

                if errors == 'all':
                    probabilities[dd-1] += (prob/3)**dd * (1-prob)**(n-dd)
                elif errors == 'X':
                    probabilities[dd-1] += prob**dd * (1-prob)**(n-dd)

                if accounted_syndromes == 2**(n-k):
                    break


    failure_rate = 1 - probability_no_error - np.cumsum(probabilities, axis=0)

    n_errors = np.array([3**i * comb(n, i) for i in range(1, n+1)])
    correctable_fraction = error_type / n_errors

    mean_iterations = np.array([
        sum((prob/3)**(dd+1) * (1-prob)**(n-dd-1) * nn * (
                sum(error_type[:dd]) + (nn+1)/2
            ) for dd, nn in enumerate(error_type[:i])
        ) for i in range(1,n+1)
    ])

    return failure_rate, correctable_fraction, mean_iterations

# def get_failure_rate_alt(self, p=None, errors='all'):
#     """Deprecated.

#     .. deprecated::
#             `get_failure_alt()` has been long deprecated. It can only be used 
#             as part of the `QGRAND()` class, it may be riddled with bugs, and 
#             it is kept here solely for reference. Use 
#             `get_weight_statistics()` instead.
#     """

#     # In this approach, the syndrome int is the inverse.

#     n, k = self.n, self.k
#     syndrome_list = np.zeros(2**(n-k), dtype=bool)
#     error_type = np.zeros(n, dtype=int)

#     stabilizers = qi.StabilizerTable.from_labels(self.stabilizers)

#     syndrome_list[0] = True
#     accounted_syndromes = 1
#     probabilities = (1-p)**n

#     for dd, error in pauli_generator(n, n+1):

#         int_syndrome = np.sum(2**np.array(stabilizers.anticommutes_with_all(error)))

#         if not syndrome_list[int_syndrome]:
#             syndrome_list[int_syndrome] = True
#             accounted_syndromes += 1

#             error_type[dd-1] += 1

#             if errors == 'all':
#                 probabilities += (p/3)**dd * (1-p)**(n-dd)
#             elif errors == 'X':
#                 probabilities += p**dd * (1-p)**(n-dd)

#             if accounted_syndromes == 2**(n-k):
#                 break

#     self.failure_rate = 1 - probabilities
#     self.correctable_errors = error_type
#     self.correctable_fraction = error_type / np.array([3**i * comb(n, i) for i in range(1, n+1)])
#     self.mean_iterations = sum((p/3)**(dd+1) * (1-p)**(n-dd-1) * nn * (sum(error_type[:dd]) + (nn+1)/2) for dd, nn in enumerate(error_type))


def t_asymp(n, D=0.3):
    return 0.5 * ((2*n + 1)*np.log(2) + np.log(np.log(2)/(1-D))) / (np.log(11.25) + 2*np.log(n))

def p_fail_asymp(error_rate, n_cx, n, D=0.3):
    return error_rate**t_asymp(n,D) * comb(n_cx, t_asymp(n,D))
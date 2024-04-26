import numpy as np
from .error import get_random_error_matrix
from .encoding import get_random_parity_check_matrix
import warnings
#from galois import GF
#from numba import njit


class GoodCode:
    """
    Determines how adequate a code is to correct errors associated with a 
    certain noise statistic. Verifies if a code is degenerate or not.

    Parameters
    ----------
    n : int
        Total number of qubits.
    k : int
        Number of qubits to protect.
    noise_probabilities : 1D array, default uniform
        (N+1) noise probabilities associated with the setup, in the same order 
        as in `error_matrix`, and including the identity. If not in decreasing
        order, they are automatically reordered. If not provided, it defaults 
        to a uniform probability.
    N : int, optional
        Number of errors. If not provided, it is inferred from either 
        `noise_probabilities` or `error_matrix`.
    error_matrix : 2D array, default random
        ((N+1) x 2n) error matrix. If not provided, then the `N` errors are 
        randomly chosen.
    parity_check_matrix : 2D array, default random
        ((n-k) x 2n) parity check matrix. If not provided, then the ``n-k`` 
        stabilizers are randomly chosen. 
    """

    def __init__(self,
                n,
                k,
                noise_probabilities = None,
                N = None,
                error_matrix = None,
                parity_check_matrix = None):

        self.n, self.k = n, k
        self.S = n-k # Number of stabilizers

        if isinstance(noise_probabilities, list):
            noise_probabilities = np.array(noise_probabilities)

        # Setting N
        if N is None:
            # 1 is subtracted to discount the identity case.
            if noise_probabilities is not None:
                self.N = noise_probabilities.shape[0] - 1
            
            elif error_matrix is not None:
                self.N = error_matrix.shape[0] - 1
        else:
            self.N = N

        # Sort in decreasing order
        if noise_probabilities is not None:
            order = np.argsort(-noise_probabilities)
            # Skip ordering if it is already ordered
            if not np.sum(order - np.arange(self.N+1)) == 0:
                noise_probabilities = noise_probabilities[order]
                error_matrix = error_matrix[order]
        else:
            noise_probabilities = np.zeros(self.N+1) + 1/(self.N+1)
        self.noise_probabilities = noise_probabilities

        if error_matrix is None:
            self.error_matrix = get_random_error_matrix(n, self.N)
        else:
            self.error_matrix = error_matrix
        assert self.error_matrix.shape[1] == 2*n, "Error matrix error!"

        if parity_check_matrix is None:
            self.parity_check_matrix = get_random_parity_check_matrix(n, self.S)
        else:
            self.parity_check_matrix = parity_check_matrix
        assert self.parity_check_matrix.shape[1] == 2*n, "Parity check matrix error!"

        self._set_syndrome_information()
        self._degen_info_is_set = False

    def _set_syndrome_information(self):
        """
        Sets syndrome related information for the class. 

        It calculates the syndrome matrix and determines unique syndromes from it. 
        Number of unique syndromes is also stored.
        """
        # Calculate the syndrome matrix by taking the product of the error matrix 
        # and the transpose of the parity check matrix, with modulo operation applied
        self.syndrome_matrix = self.error_matrix @ self.parity_check_matrix.T % 2
        
        # Get unique syndromes preserving the original order
        # See https://stackoverflow.com/a/12926989
        indices = np.unique(self.syndrome_matrix, axis=0, return_index=True)[1]
        self.unique_syndrome_matrix = self.syndrome_matrix[np.sort(indices)]
        
        # Store the number of unique syndromes
        self.n_unique_syndromes = self.unique_syndrome_matrix.shape[0]

    def _set_degeneracy_information(self):
        """
        Sets degeneracy related information for the class.

        It computes product of errors and the indices for tuples and singles 
        if they have not been set before.
        """
        if not self._degen_info_is_set:
            self.prod_errors, self.tuple_idx, self.single_idx = self._get_degeneracy_information(
                self.syndrome_matrix,
                self.unique_syndrome_matrix,
                self.error_matrix
            )
            self._degen_info_is_set = True

    @staticmethod
    def _get_degeneracy_information(syndrome_matrix,
                                    unique_syndrome_matrix,
                                    error_matrix):
        """
        Computes the degeneracy information.

        Parameters
        ----------
        syndrome_matrix : numpy.ndarray
            The matrix representing syndromes.
        unique_syndrome_matrix : numpy.ndarray
            The matrix with unique syndromes.
        error_matrix : numpy.ndarray
            The matrix representing errors.

        Returns
        -------
        prod_errors : numpy.ndarray
            Product of error pairs.
        tuple_idx : numpy.ndarray
            Indices of tuples in the syndrome matrix.
        single_idx : numpy.ndarray
            Indices of singles in the syndrome matrix.
        """
        n_unique_syndromes = unique_syndrome_matrix.shape[0]
        n = error_matrix.shape[1] // 2

        # Initialize lists to store indices of errors with same syndrome
        repeated_idx = []
        single_idx = []
        
        # Separate indices of unique and repeated syndromes
        for syndrome in unique_syndrome_matrix:
            pos = np.squeeze(np.argwhere(np.all(syndrome_matrix == syndrome, axis=1)).ravel())
            if pos.size == 1:
                single_idx.append(pos)
            else:
                repeated_idx.append(pos)
            
        single_idx = np.array(single_idx)
        
        # Calculate pairs of errors with the same syndrome
        n_pairs = syndrome_matrix.shape[0] - n_unique_syndromes
        tuple_idx = np.zeros((n_pairs, 2), dtype=int)
        s = 0
        for idx in repeated_idx:
            for ind in idx[1:]:
                tuple_idx[s] = idx[0], ind
                s += 1
        
        # Assert that the number of pairs equals the expected number
        assert (error_matrix.shape[0] - n_unique_syndromes) == tuple_idx.shape[0], "Error!"

        # Calculate the product of these error pairs
        prod_errors = (error_matrix[tuple_idx[:, 0]] + error_matrix[tuple_idx[:, 1]]) % 2

        # Swapping Z and X halves of errors, for comparison with parity check matrix.
        prod_errors = np.roll(prod_errors, n, axis=1)

        return prod_errors, tuple_idx, single_idx

    def check_if_good_code(self):
        """
        Check if the code is robust to the noise statistics, and can correct 
        all present noise.

        Returns
        -------
        bool
            Whether or not the code is good.
        """

        is_nondegenerate = self._check_if_nondegenerate(self.syndrome_matrix,
                                                self.unique_syndrome_matrix)

        if is_nondegenerate:
            return True

        # Getting prod_errors if it is not computed yet.
        self._set_degeneracy_information()        
        is_degenerate = self._check_if_degenerate(self.parity_check_matrix,
                                                self.prod_errors,
                                                iterative = False)

        return is_degenerate

    @staticmethod
    def _check_if_nondegenerate(syndrome_matrix,
                                unique_syndrome_matrix=None):
        """
        Check if a code is nondegenerate with regards to some noise 
        statistics, by verifying if syndrome matrix has any duplicate rows. If 
        there are no duplicate rows, then there is a one-to-one mapping 
        between the syndromes and the errors, so the code is good, as it can 
        correct all errors in the noise statistics.

        Parameters
        ----------
        syndrome_matrix : 2D array
            `((N+1) x S)` syndrome matrix.

        Returns
        -------
        bool
            Whether or not the code is nondegenerate (regarding some specific 
            noise statistics).

        Notes
        -----
        A previous method to compute nondegeneracy was python-based::
        
            #From https://stackoverflow.com/questions/26165723/how-to-test-if-all-rows-are-distinct-in-numpy
            return len(set(tuple(zip(*syndrome_matrix.T)))) == len(syndrome_matrix)
        
        """

        if unique_syndrome_matrix is None:
            unique_syndrome_matrix = np.unique(syndrome_matrix, axis=0)

        is_nondegenerate = unique_syndrome_matrix.shape[0] == syndrome_matrix.shape[0]
        return is_nondegenerate        

    @staticmethod
    def _check_if_degenerate(parity_check_matrix,
                            prod_errors,
                            iterative = False,
                            get_stabilizer_info = False):
        """
        Checks if a degenerate code is good.

        Parameters
        ----------
        parity_check_matrix : 2D array
            (S x 2n) parity check matrix.
        error_matrix : 2D array
            (P x 2n) product error matrix. Check `_get_product_errors()` for 
            details.
        iterative : bool, default False
            Whether to test the errors iteratively, one by one, or at once, 
            as the full matrix. If N is small, the matrix approach may be more 
            efficient.

        Returns
        -------
        bool
            If ``True``, it is a good degenerate code, and if ``False``, it is
            not a good code, that is, there is some error in the noise 
            statistics that the code cannot correct.
        """

        is_degenerate = True # Assume it is good

        if iterative:
            for error in prod_errors:
                # For each error product, check if it is a (not necessarily 
                # minimal) stabilizer. If Yes, then the two errors have equivalent
                # effects on the circuit.
                if not GoodCode._get_boolean_solution(parity_check_matrix.T, error):
                    # If not a stabilizer, then the code is not good
                    is_degenerate = False
                    break
            return is_degenerate

        else:
            is_stabilizer = GoodCode._get_boolean_solution(parity_check_matrix.T, prod_errors)
            if not np.all(is_stabilizer):
                is_degenerate = False

            if get_stabilizer_info:
                return is_degenerate, is_stabilizer
            else:
                return is_degenerate
            
    @staticmethod
    def _get_boolean_solution(m, error):
        """
        Checks if the error is a (not necessarily minimal) stabilizer, with 
        regards to the parity check matrix.

        Implements Gauss-Jordan elimination for boolean matrices, since that 
        functionality doesn't seem to exist in numpy.

        Parameters
        ----------
        m : 2D array
            Transpose of S x 2n parity check matrix.
        error : 1D array, 2D array
            2n sized error, or (P x 2n) product error matrix. The format is in 
            [X|Z], and not the usual [Z|X].

        Returns
        -------
        is_stabilizer : bool, 1D array
            If `error` is a 1D array, returns whether that array is a 
            stabilizer. If it is a 2D array, returns a P sized 1D bool array.

        Notes
        -----
        Because all S minimal stabilizers are linearly independent, and 
        ``S < 2*n`` always, the rank of the matrix `m` is necessarily S. 
        Therefore, in row echelon form, the matrix `m` should have only ones 
        in its diagonal. Consequently, if the error column has any 1s past the 
        first S entries, then the system is unsolvable, and the error is not 
        a stabilizer. Otherwise, a solution exists, and error is a stabilizer.

        Warnings
        --------
        The `error` argument should be given in the form [X|Z], and not [Z|X].
        """

        if len(error.shape) == 1:
            only_one_error = True
            error = error[:,None]
        else:
            only_one_error = False
            error = error.T

        M = np.c_[m, error]
        
        for i in range(m.shape[1]):
            col = M[i:,i]
            rows_to_change = np.nonzero(col)[0] + i

            # Should never occur except at the end, since the stabilizers are 
            # linearly independent.
            if rows_to_change.shape[0] == 0:
                continue
            M[rows_to_change[1:]] -= M[rows_to_change[0]]
            M = M % 2
            if rows_to_change[0] != i:
                M[[i, rows_to_change[0]]] = M[[rows_to_change[0], i]]
        
        is_stabilizer = ~np.any(M[m.shape[1]:, m.shape[1]:], axis=0)
        
        if only_one_error:
            return is_stabilizer[0]
        else:
            return is_stabilizer

    @staticmethod
    def _get_boolean_solution_2(m, error):
        """
        Checks if the error is a (not necessarily minimal) stabilizer, with 
        regards to the parity check matrix.

        Implements Gauss-Jordan elimination for boolean matrices, since that 
        functionality doesn't seem to exist in numpy.

        Parameters
        ----------
        m : 2D array
            Transpose of S x 2n parity check matrix.
        error : 1D array, 2D array
            2n sized error, or (P x 2n) product error matrix. The format is in 
            [X|Z], and not the usual [Z|X].

        Returns
        -------
        is_stabilizer : bool, 1D array
            If `error` is a 1D array, returns whether that array is a 
            stabilizer. If it is a 2D array, returns a P sized 1D bool array.

        Notes
        -----
        Because all S minimal stabilizers are linearly independent, and 
        ``S < 2*n`` always, the rank of the matrix `m` is necessarily S. 
        Therefore, in row echelon form, the matrix `m` should have only ones 
        in its diagonal. Consequently, if the error column has any 1s past the 
        first S entries, then the system is unsolvable, and the error is not 
        a stabilizer. Otherwise, a solution exists, and error is a stabilizer.

        Warnings
        --------
        The `error` argument should be given in the form [X|Z], and not [Z|X].
        """

        error = error[:,None]

        M = np.c_[m, error]
        M = M.T
        
        M = GF(2)(M)
        rank = np.linalg.matrix_rank(M)

        is_stabilizer = rank!=M.shape[0]
        
        return is_stabilizer

    @staticmethod
    #@njit()
    def _get_boolean_solution_3(rre_parity_check_matrix, pivots,
                                error):
        """
        Checks if the error is a (not necessarily minimal) stabilizer, with 
        regards to the parity check matrix.

        Implements Gauss-Jordan elimination for boolean matrices, since that 
        functionality doesn't seem to exist in numpy.

        Parameters
        ----------
        m : 2D array
            Transpose of S x 2n parity check matrix.
        error : 1D array, 2D array
            2n sized error, or (P x 2n) product error matrix. The format is in 
            [X|Z], and not the usual [Z|X].

        Returns
        -------
        is_stabilizer : bool, 1D array
            If `error` is a 1D array, returns whether that array is a 
            stabilizer. If it is a 2D array, returns a P sized 1D bool array.

        Notes
        -----
        Because all S minimal stabilizers are linearly independent, and 
        ``S < 2*n`` always, the rank of the matrix `m` is necessarily S. 
        Therefore, in row echelon form, the matrix `m` should have only ones 
        in its diagonal. Consequently, if the error column has any 1s past the 
        first S entries, then the system is unsolvable, and the error is not 
        a stabilizer. Otherwise, a solution exists, and error is a stabilizer.

        Warnings
        --------
        The `error` argument should be given in the form [X|Z], and not [Z|X].
        """
        
        i = 0
        col_size = rre_parity_check_matrix.shape[1] 
        while i < col_size:
            if error[i] == 1:
                ind = pivots[i]
                if ind == -1:
                    return False
                else:
                    error = (error + rre_parity_check_matrix[ind]) % 2
            i += 1
        return True

    def get_success_rate(self):
        self._set_degeneracy_information()
        return self._get_success_rate(self.parity_check_matrix,
                                        self.prod_errors,
                                        self.tuple_idx,
                                        self.single_idx,
                                        self.noise_probabilities)

    @staticmethod
    def _get_success_rate(parity_check_matrix,
                        prod_errors,
                        tuple_idx,
                        single_idx,
                        noise_probabilities):
        """
        Compute the success rate of the code

        Parameters
        ----------
        syndrome_matrix : 2D array
            ((N+1) x S) syndrome matrix.
        parity_check_matrix : 2D array
            (S x 2n) parity check matrix.
        error_matrix : 2D array
            ((N+1) x 2n) error matrix.

        Returns
        -------
        int
            If equal to 2, the code is actually nondegenerate. If equal to 1, 
            it is a good degenerate code, and if 0, it is not a good code, 
            that is, there is some error in the noise statistics that the code
            cannot correct.
        """

        is_stabilizer = GoodCode._get_boolean_solution(parity_check_matrix.T, prod_errors)

        # Get which errors are indeed correctable
        n_errors = noise_probabilities.shape[0]
        is_correctable = np.zeros(n_errors, dtype=int)
        is_correctable[np.unique(tuple_idx[:,0])] = 1 # coset leaders
        is_correctable[tuple_idx[:,1]] = is_stabilizer.astype(int)
        is_correctable[single_idx] = 1

        # Compute success rate
        success_rate = np.sum(is_correctable * noise_probabilities)
        
        return success_rate

    def get_success_profile(self):
        """
        Compute the success rate of the code

        Parameters
        ----------
        syndrome_matrix : 2D array
            ((N+1) x S) syndrome matrix.
        parity_check_matrix : 2D array
            (S x 2n) parity check matrix.
        error_matrix : 2D array
            ((N+1) x 2n) error matrix.

        Returns
        -------
        int
            If equal to 2, the code is actually nondegenerate. If equal to 1, 
            it is a good degenerate code, and if 0, it is not a good code, 
            that is, there is some error in the noise statistics that the code
            cannot correct.
        """

        is_nondegenerate = self._check_if_nondegenerate(self.syndrome_matrix,
                                                self.unique_syndrome_matrix)

        if not is_nondegenerate:
            # Getting prod_errors if it is not computed yet.
            self._set_degeneracy_information()        
            is_degenerate, is_stabilizer = self._check_if_degenerate(
                                                self.parity_check_matrix,
                                                self.prod_errors,
                                                iterative = False,
                                                get_stabilizer_info = True)

            # Get which errors are indeed correctable
            n_errors = self.noise_probabilities.shape[0]
            is_correctable = np.zeros(n_errors, dtype=int)
            is_correctable[np.unique(self.tuple_idx[:,0])] = 1 # coset leaders
            is_correctable[self.tuple_idx[:,1]] = is_stabilizer.astype(int)
            is_correctable[self.single_idx] = 1

            # Compute success rate
            success_rate = np.sum(is_correctable * self.noise_probabilities)
            good_code = is_degenerate

        else:
            success_rate = 1.
            good_code = True
            is_degenerate = False
        
        return good_code, is_nondegenerate, is_degenerate, success_rate

        
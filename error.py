"""Functions for the errors.

Defines functions related to the errors in the code.
"""

import numpy as np
from scipy.special import comb
from .pauli import pauli_generator

### STATISTICS

def get_bernoulli_statistics(n, p, bits=None, n_types=3):
    """
    Gets the probabilities associated with bernoulli type errors, where each 
    qubit can suffer a particular error with probability `p`.

    Parameters
    ----------
    n : int
        Number of qubits.
    p : float
        Probability of some qubit suffering a specific error. The probability 
        of it suffering any error is `n_types x p`.
    bits : int, default n
        Cutoff for computing the probabilities. They are computed up to `bits`
        qubits.
    n_types : int, default 3
        Number of error types assumed, defaults to 3 to account for X, Y, and 
        Z type errors.

    Returns
    -------
    1D array
        Probability distribution.
    """

    if bits is None:
        bits = n

    n_errors = 1 + sum(
        comb(n, b, exact=True) * n_types**b for b in range(1, bits+1)
    )

    probs = np.zeros(n_errors)
    ind_0 = 0
    for b in range(0, bits+1):
        n_error = comb(n, b, exact=True) * n_types**b
        probs[ind_0: ind_0+n_error] = p**b * (1-p)**(n-b)

    return probs

### GENERATION

def get_random_error_matrix(n, N):
    """
    Creates a random 2n x (N+1) error matrix of distinct errors. It 
    includes the identity case as the first row.

    Parameters
    ----------
    n : int
        Number of qubits.
    N : int
        Number of errors.

    Returns
    -------
    2D array
        2n x (N+1) error matrix. It includes the identity case as the 
        first row.
    """

    g = np.random.default_rng()
    error_matrix = g.choice(4**n - 1, size=N+1, replace=False) + 1

    # Setting first row as identity case
    error_matrix[0] = 0

    # From https://stackoverflow.com/q/22227595
    error_matrix = (
        ((error_matrix[:,None] & (1 << np.arange(2*n))[::-1])) > 0
    ).astype(int)

    return error_matrix

def get_general_error_matrix(n, bits=1, force=False):
    """
    Generates the error matrix associated to bernoulli errors affecting up to 
    `bits` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    bits : int, default 1
        Maximum number of qubits affected that is included.
    force : bool, default False
        Whether to force creation of the error matrix even when the number of 
        errors surpasses one million.

    Returns
    -------
    error_matrix: 2D array
        (N x 2n) error matrix. 
    """

    n_errors = sum(comb(n, b, exact=True) * 3**b for b in range(1, bits+1))
    assert n_errors < 1_000_000 and not force,\
        ('There are {} errors. '.format(n_errors) +\
        'If you wish to compute the whole table, set force=True.')

    possible_errors = list(pauli_generator(n, bits+1))
    assert n_errors == len(possible_errors)    

    error_matrix = np.zeros((n_errors, 2*n), dtype=int)

    for i, (_, error) in enumerate(possible_errors):
        for j, pauli in enumerate(list(error)):
            if pauli == 'I':
                continue
            else:
                if pauli != 'X':
                    error_matrix[i, j] = 1
                if pauli != 'Z':
                    error_matrix[i, n+j] = 1

    return error_matrix

def get_error_matrix(noise):
    """Gets error matrix from error list in Qiskit format.

    Parameters
    ----------
    noise : list
        List of errors, in Qiskit string format. The order in which they are 
        given is preserved for the error matrix.

    Returns
    -------
    2D array
        (len(noise) x 2n) 2D array error matrix.    
    """

    n_errors = len(noise)
    n = len(noise[0])
    error_matrix = np.zeros((n_errors, 2*n), dtype=int)

    for i, error in enumerate(noise):
        for j, pauli in enumerate(list(error)):
            if pauli == 'I':
                continue
            else:
                if pauli != 'X':
                    error_matrix[i, j] = 1
                if pauli != 'Z':
                    error_matrix[i, n+j] = 1

    return error_matrix

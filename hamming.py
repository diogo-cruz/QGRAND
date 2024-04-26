"""Functions for the code's Hamming bound.

Defines functions related to computing the Hamming bound for the code.
"""

import numpy as np
from scipy.special import comb

def shannon_entropy_bernoulli(p):
    """
    Compute the Shannon entropy for a Bernoulli distribution in base 2.

    Parameters
    ----------
    p : float
        Probability of the success in the Bernoulli distribution.

    Returns
    -------
    float
        Shannon entropy of the Bernoulli distribution.
    """
    
    # Avoid issues with log2(0) and replace NaNs with zeros
    entropy = -np.nan_to_num(p * np.log2(p)) - np.nan_to_num((1 - p) * np.log2(1 - p))
    
    return entropy


def hamming_bound(x):
    """
    Compute the Hamming bound.

    Parameters
    ----------
    x : float
        The proportion of error bits.

    Returns
    -------
    float
        The Hamming bound.
    """

    # Return the Hamming bound by using the Shannon entropy for a Bernoulli distribution
    return 1 - x * np.log2(3) - shannon_entropy_bernoulli(x)


def asympt_max_distance_hamming(n, k, hamming_f):
    """
    Compute the asymptotic maximum Hamming distance.

    Parameters
    ----------
    n : int
        Length of the codeword.
    k : int
        Length of the message.
    hamming_f : callable
        Function that computes the Hamming bound.

    Returns
    -------
    int
        The asymptotic maximum Hamming distance.
    """

    # Return the asymptotic max distance considering the Hamming bound
    return int(2 * n * hamming_f(k / n) + 1)


def asympt_max_distance_varshamov(n, k, hamming_f):
    """
    Compute the asymptotic maximum Varshamov distance.

    Parameters
    ----------
    n : int
        Length of the codeword.
    k : int
        Length of the message.
    hamming_f : callable
        Function that computes the Hamming bound.

    Returns
    -------
    int
        The asymptotic maximum Varshamov distance.
    """

    # Return the asymptotic max distance considering the Hamming bound
    return int(n * hamming_f(k / n))


def max_distance_hamming(n, k, hamming_f=None):
    """
    Compute the maximum Hamming distance. If the result exceeds the numeric
    limits, it resorts to the asymptotic computation.

    Parameters
    ----------
    n : int
        Length of the codeword.
    k : int
        Length of the message.
    hamming_f : callable, optional
        Function that computes the Hamming bound.

    Returns
    -------
    int
        The maximum Hamming distance.
    """

    try:
        s = 0.0  # Initialize the sum
        
        # Iterate over a range and compute a cumulative sum
        for t in range(int((n - k) / 2) + 1):
            s += comb(n, t) * 3**t / 2**(n - k)
            if s > 1:
                return 2 * (t - 1) + 1
    except OverflowError:
        # If an OverflowError occurs, resort to the asymptotic computation
        return asympt_max_distance_hamming(n, k, hamming_f)

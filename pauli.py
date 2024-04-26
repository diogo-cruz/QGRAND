"""Auxiliary functions to parse Pauli strings.

"""

import numpy as np
from scipy.special import comb
from itertools import combinations, product


### AUXILIARY

def get_numbered_noise(noise):
    """Convert Pauli strings in Qiskit's to Openfermion's format.

    Parameters
    ----------
    noise : str
        Pauli string in Qiskit format (for example, ``'XIZII'``).

    Returns
    -------
    str
        Pauli string in Openfermion format (for example, 'X0 Z2').

    Warnings
    --------
    Note that the order may not be correct, as Qiskit and Openfermion do not 
    necessarily use the same endianness. I haven't confirmed it is correct.    
    """

    numbered_noise = []

    for i, op in enumerate(list(noise)):
        if op == 'I':
            continue
        else:
            numbered_noise.append(op + str(i))

    return ' '.join(numbered_noise)

def pauli_generator(n, distance_max):
    """Generator for Pauli strings in order of increasing weight.

    Generates Pauli strings for `n` qubits, starting with the strings having 
    only one non-identity entry (with each entry running through X, Y, and Z, 
    in this order), and continuing until the last string with `distance_max-1` 
    non-identity entries.

    Parameters
    ----------
    n : int
        Number of qubits.
    distance_max : int
        Generates strings with up to `distance_max-1` non-identity entries. 
        If using this function to compute the distance, it is the maximum 
        distance considered.

    Yields
    ------
    int
        Number of non-identity indices for the yielded string.
    str
        Pauli string.
    """

    array = ['I' for i in range(n)]
    paulis = ['X', 'Y', 'Z']
    for dd in range(1, distance_max):
        combos = combinations(range(n), dd)
        all_ops = product(paulis, repeat=dd)
        errors = product(all_ops, combos)
        for ops, inds in errors:
            for op, ind in zip(ops, inds):
                array[ind] = op
            str_array = ''.join(array)
            yield dd, str_array
            for ind in inds:
                array[ind] = 'I'

def pauli_generator_num(n, distance_max):
    """Generator for Pauli strings in order of increasing weight, in binary.

    Generates Pauli strings for `n` qubits, starting with the strings having 
    only one non-identity entry (with each entry running through X, Y, and Z, 
    in this order), and continuing until the last string with `distance_max-1` 
    non-identity entries. Uses the binary representation, as seen in the rows 
    of the parity check matrix.

    Parameters
    ----------
    n : int
        Number of qubits.
    distance_max : int
        Generates strings with up to `distance_max-1` non-identity entries. 
        If using this function to compute the distance, it is the maximum 
        distance considered.

    Yields
    ------
    int
        Number of non-identity indices for the yielded string.
    1D array
        Pauli string, in binary form, as a 2n-sized 1D array.
    """

    array = np.zeros(2*n, dtype=int)
    paulis = ['X', 'Y', 'Z']
    for dd in range(1, distance_max):
        combos = combinations(range(n), dd)
        all_ops = product(paulis, repeat=dd)
        errors = product(all_ops, combos)
        for ops, inds in errors:
            for op, ind in zip(ops, inds):
                if op == 'I':
                    continue
                if op != 'X':
                    array[ind] = 1
                if op != 'Z':
                    array[n+ind] = 1
            yield dd, array
            # for ind in inds:
            #     array[ind] = 0
            #     array[n+ind] = 0
            inds = np.array(inds)
            array[inds] = 0
            array[n+inds] = 0

def pauli_generator_old(n, distance_max):
    """Deprecated. Generator for Pauli strings.

    .. deprecated:: 1.0
            `pauli_generator_old()` is non-functional, `pauli_generator_num()` 
            should be used instead.

    Generates Pauli strings for `n` qubits, starting with the strings having 
    only one non-identity entry (with each entry running through X, Y, and Z, 
    in this order), and continuing until the last string with `distance_max-1` 
    non-identity entries. Uses the binary representation, as seen in the rows 
    of the parity check matrix.

    Parameters
    ----------
    n : int
        Number of qubits.
    distance_max : int
        Generates strings with up to `distance_max-1` non-identity entries. 
        If using this function to compute the distance, it is the maximum 
        distance considered.

    Yields
    ------
    int
        Number of non-identity indices for the yielded string.
    1D array
        Pauli string, in binary form, as a 2n-sized 1D boolean array.

    Warnings
    --------
    It is not clear that this function correctly computes the Pauli strings, 
    as some issues may have only been fixed in `pauli_generator_num()`.

    """

    array = np.array([False] * 2*n, dtype=bool)
    for dd in range(1, distance_max):
        for combo in combinations(range(n), dd):
            combo = np.array(combo)
            array[combo] = True
            #print(combo, array)
            yield dd, array
            array[combo+n] = True
            #print(combo, array)
            yield dd, array
            array[combo] = False
            #print(combo, array)
            yield dd, array
            array[combo+n] = False
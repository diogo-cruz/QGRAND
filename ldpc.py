import numpy as np
from .GoodCode import GoodCode
from time import time

def get_random_QLDPC_parity_check_matrix(n,
                                        k,
                                        c,
                                        timeout=300,
                                        sanity_check=True):
    """
    Generates a random QLDPC code, by generating its parity check matrix.

    Parameters
    ----------
    n : int
        Number of qubits in the encoding.
    k : int
        Original number of data qubits.
    c : int
        Maximum number of qubits on which each minimal stabilizer can act 
        nontrivially.
    timeout : int, default 300
        Maximum time, in seconds, allotted to generating the code. This is 
        necessary since it may occasionally get stuck at a particularly bad 
        attempt.
    sanity_check : bool, default True
        Whether or not to check if the minimal stabilizers generated mutually 
        commute and are linearly independent. This should always be the case, 
        unless something went wrong.
    
    Returns
    -------
    parity_check_matrix : 2D array
        ((n-k) x 2n) 2D binary array, representing the parity check matrix of 
        the code.

    Notes
    -----
    This generation should scale roughly, at worst, with O(n^3) (or possibly 
    O(n^4)). The two main bottlenecks are the luck we get at finding suitable 
    Pauli strings for new minimal stabilizers (in the non-LDPC case this is O(1)
    at each iteration, not sure what it is here), and the runtime of checking 
    whether the new string is linearly independent, which may scale as O(n^3) in
    the worst case, though the average is much better.

    See `QLDPC.ipynb` for earlier versions of this code.
    """

    start = time()

    s = n-k
    H_X = np.zeros((s, n), dtype=int)
    H_Z = np.zeros((s, n), dtype=int)

    # First minimal stabilizer
    pos = np.sort(np.random.choice(n, size=c, replace=False))
    op_X = np.random.randint(2, size=c)
    op_Z = np.random.randint(2, size=c)
    while np.sum(op_X + op_Z) == 0: # Ensuring it is not the identity
        op_X = np.random.randint(2, size=c)
        op_Z = np.random.randint(2, size=c)
    H_X[0, pos] = op_X
    H_Z[0, pos] = op_Z
        
    for si in range(1, s):
        while True:
            pos = np.sort(np.random.choice(n, size=c, replace=False))
            op_X = np.random.randint(2, size=c)
            op_Z = np.random.randint(2, size=c)
            while np.sum(op_X + op_Z) == 0: # Ensuring it is not the identity
                op_X = np.random.randint(2, size=c)
                op_Z = np.random.randint(2, size=c)
            pos_X = pos[op_X.astype(bool)]
            pos_Z = pos[op_Z.astype(bool)]
            # Only accept if new Pauli strings commutes with previous stabilizers
            if commutes_3(H_Z[:si, pos_X], H_X[:si, pos_Z]):
                H_non_trivial = np.concatenate((H_X[:si], H_Z[:si]), axis=1)
                new_Hi = np.zeros(2*n, dtype=int)
                new_Hi[pos_X] = 1
                new_Hi[n+pos_Z] = 1
                # Only accept if new string is linearly independent
                if is_new_minimal_stabilizer(new_Hi, H_non_trivial):
                    H_X[si, pos_X] = 1
                    H_Z[si, pos_Z] = 1
                    break
            middle = time()
            if middle-start > timeout:
                print("Timeout! Retrying...")
                return get_random_QLDPC_parity_check_matrix(n,
                                        k,
                                        c,
                                        timeout=timeout,
                                        sanity_check=sanity_check)

    parity_check_matrix = np.concatenate((H_X, H_Z), axis=1)

    if sanity_check:
        sA = stabilizers_commute(H_X, H_Z)
        sB = stabilizers_are_linearly_independent(parity_check_matrix)
        assert sA and sB, "Error!"
    
    return parity_check_matrix

def stabilizers_commute(H_X, H_Z):
    """
    Checks if two stabilizer operators commute.

    Parameters
    ----------
    H_X : np.ndarray
        A representation of the first stabilizer operator.
    H_Z : np.ndarray
        A representation of the second stabilizer operator.

    Returns
    -------
    bool
        True if the operators commute, False otherwise.
    """
    # Compute the commutator of the two operators and check if it's zero
    commutator = (H_X @ H_Z.T + H_Z @ H_X.T) % 2
    return not bool(np.sum(commutator))

def stabilizers_are_linearly_independent(H):
    """
    Checks if a set of stabilizers is linearly independent.

    Parameters
    ----------
    H : np.ndarray
        Matrix where each row represents a stabilizer.

    Returns
    -------
    bool
        True if the stabilizers are linearly independent, False otherwise.
    """
    s = H.shape[0]
    # For each stabilizer, check if it's new and minimal in the stabilizer space
    return all(is_new_minimal_stabilizer(H[i], H[[j for j in range(s) if j!=i]]) for i in range(s))

def stabilizer_weight(H):
    """
    Calculates the weight of the stabilizer operators.

    Parameters
    ----------
    H : np.ndarray
        Matrix where each row represents a stabilizer.

    Returns
    -------
    np.ndarray
        A list with the weights of each stabilizer.
    """
    n = H.shape[1]//2
    # The weight is the number of non-identity operators
    return np.sum((H[:,:n] + H[:,n:]).astype(bool).astype(int), axis=1)

def is_new_minimal_stabilizer(tentative_stabilizer, parity_check_matrix):
    """
    Checks if a stabilizer is new and minimal in the stabilizer space.

    Parameters
    ----------
    tentative_stabilizer : np.ndarray
        The stabilizer to be checked.
    parity_check_matrix : np.ndarray
        The parity check matrix.

    Returns
    -------
    bool
        True if the stabilizer is new and minimal, False otherwise.
    """
    return not GoodCode._get_boolean_solution(parity_check_matrix.T, tentative_stabilizer)

def commutes(op_1, op_M):
    """
    Checks if two operators commute.

    Parameters
    ----------
    op_1 : np.ndarray
        The first operator.
    op_M : np.ndarray
        The second operator.

    Returns
    -------
    bool
        True if the operators commute, False otherwise.
    """
    return np.all(1 - ((op_M @ op_1) % 2))

def commutes_2(op_1, op_M_1, op_2, op_M_2):
    """
    Checks if two pairs of operators commute.

    Parameters
    ----------
    op_1 : np.ndarray
        The first operator in the first pair.
    op_M_1 : np.ndarray
        The second operator in the first pair.
    op_2 : np.ndarray
        The first operator in the second pair.
    op_M_2 : np.ndarray
        The second operator in the second pair.

    Returns
    -------
    bool
        True if the pairs commute, False otherwise.
    """
    return np.all(1 - ((op_M_1 @ op_1 + op_M_2 @ op_2) % 2))

def commutes_3(op_M_1, op_M_2):
    """
    Checks if the sum of each pair of operators commutes.

    Parameters
    ----------
    op_M_1 : np.ndarray
        The first set of operators.
    op_M_2 : np.ndarray
        The second set of operators.

    Returns
    -------
    bool
        True if the sums commute, False otherwise.
    """
    return np.all(1 - ((np.sum(op_M_2, axis=1) + np.sum(op_M_1, axis=1)) % 2))

def pauli(H_X, H_Z):
    """
    Converts a given set of operators into their Pauli representation.

    Parameters
    ----------
    H_X : np.ndarray
        The set of X operators.
    H_Z : np.ndarray
        The set of Z operators.

    Returns
    -------
    list
        A list of strings, where each string represents an operator in the Pauli representation.
    """
    # Define the map from the integer representation to Pauli representation
    mat = {0:'I',1:'X',2:'Z',3:'Y'}
    H = H_X + 2*H_Z
    pauli_list = []

    # Loop through each row of the operators
    for row in H:
        # Start the string representation with the '+' sign
        op = '+'
        # Add the Pauli representation of each element in the row
        for col in row:
            op += mat[col]
        pauli_list.append(op)
    
    return pauli_list

def pauli_H(H):
    """
    Converts a given set of operators into their Pauli representation.

    Parameters
    ----------
    H : np.ndarray
        The set of operators.

    Returns
    -------
    list
        A list of strings, where each string represents an operator in the Pauli representation.
    """
    n = H.shape[1]//2
    return pauli(H[:,:n], H[:,n:])
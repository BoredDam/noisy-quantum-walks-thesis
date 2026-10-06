import numpy as np
from scipy import sparse

def H():
    H = [[1, 1], [1, -1]]
    H = np.array(H)
    H = H / np.sqrt(2)
    return H


def X():
    X = [[0, 1], [1, 0]]
    X = np.array(X)
    return X


def Z():
    Z = [[1, 0], [0, -1]]
    Z = np.array(Z)
    return Z


def custom_H(p):
    H = [[np.sqrt(p), np.sqrt(1 - p)], 
         [np.sqrt(1 - p), -np.sqrt(p)]]
    H = np.array(H)
    return H


def Y():
    Y = [[1, 1j], [1j, 1]]
    Y = np.array(Y)
    Y = Y / np.sqrt(2)
    return Y


def coin_hilbert(symmetrical: bool = True):
    C = np.array([[1], [0]])
    if symmetrical:
        C = np.matmul(Y(), C)
    return C

def _S_circle(N: int):
    """
    builds the matrix for the shift operator S
    of the given size `N` for the random walk on the circle
    """
    i = np.arange(N)

    indices = np.concatenate((
        (i - 1) % N,
        N + (i + 1) % N
    ))

    indptr = np.arange(2*N + 1)
    data = np.ones(2*N)

    return sparse.csr_matrix(
        (data, indices, indptr),
        shape=(2*N, 2*N)
    )

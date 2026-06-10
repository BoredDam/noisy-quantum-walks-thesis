import numpy as np


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

    A = np.zeros([N, N])
    B = np.zeros([N, N])

    for i in range(N):
        a1 = np.full([N, 1], 0)
        a2 = np.full([1, N], 0)

        a1[(i + 1) % N] = 1
        a2[0][i] = 1
        A += a1 @ a2

    for i in range(N):
        b1 = np.full([N, 1], 0)
        b2 = np.full([1, N], 0)
        b1[i - 1 % N][0] = 1
        b2[0][i] = 1
        B += b1 @ b2

    c = np.array([[1, 0], [0, 0]])
    d = np.array([[0, 0], [0, 1]])

    S = np.kron(c, A) + np.kron(d, B)
    return S
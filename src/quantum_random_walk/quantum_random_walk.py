import numpy as np
from quantum_utils import X, Y, Z, H, custom_H


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


def circle_quantum_random_walk_1D(
        N: int, 
        steps: int, 
        odds: float = 0.5, 
        symmetrical: bool = True
        ):

    I = np.identity(N)
    S = _S_circle(N)
    H = custom_H(odds)
    U = S @ np.kron(H, I)

    start = np.full([N, 1], 0)
    start[N // 2] = 1
    vec = np.kron(coin_hilbert(symmetrical), start)

    data = []
    for _ in range(steps):
        current_vec = vec.reshape(2, N)
        prob_pos = np.sum((np.abs(current_vec)) ** 2, axis=0)
        vec = U @ vec
        data.append(prob_pos)

    return data


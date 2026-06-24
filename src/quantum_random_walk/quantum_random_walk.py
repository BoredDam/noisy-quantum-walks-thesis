import numpy as np
from src.quantum_random_walk.quantum_utils import X, Y, Z, H, custom_H, _S_circle, coin_hilbert


def circle_quantum_random_walk_1D(
        max_pos: int, 
        steps: int, 
        odds: float = 0.5, 
        symmetrical: bool = True
        ):

    N = 2*max_pos + 1
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

    return data, np.array(range(-N//2, N//2))


def noisy_circle_quantum_random_walk_1D(
        max_pos: int, 
        steps: int, 
        odds: float = 0.5, 
        symmetrical: bool = True, 
        noise_type: str = None,
        noise_odds: float = 0.02,
        seed: int = None
    ):

    N = 2*max_pos + 1
    I = np.identity(N)
    S = _S_circle(N)
    H = custom_H(odds)
    U = S @ np.kron(H, I)

    if seed != None:
        np.random.seed(seed)

    match noise_type:
        case "X":
            noise_mtx = np.array([[0, 1], [1, 0]])
        case "Z":
            noise_mtx = np.array([[1, 0], [0, -1]])
        case "H":
            noise_mtx = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
        case None:
            noise_mtx = np.identity(2)
    noise_operator = np.kron(noise_mtx, I)

    start = np.full([N, 1], 0)
    start[N // 2] = 1
    vec = np.kron(coin_hilbert(symmetrical), start)

    data = []
    for _ in range(steps):
        current_vec = vec.reshape(2, N)
        prob_pos = np.sum((np.abs(current_vec)) ** 2, axis=0)
        if np.random.rand() <= noise_odds:
            vec = noise_operator @ U @ vec
        else:
            vec = U @ vec
        data.append(prob_pos)

    return data, np.array(range(-N//2, N//2))


def noisy_meas_circle_quantum_random_walk_1D(
        max_pos: int, 
        steps: int, 
        odds: float = 0.5, 
        symmetrical: bool = True, 
        meas_odds: float = 0.02,
        seed: int = None
    ):

    N = 2*max_pos + 1
    I = np.identity(N, dtype=complex)
    S = _S_circle(N)
    H = custom_H(odds)

    U = S @ np.kron(H, I)
    if seed != None:
        np.random.seed(seed)

    start = np.zeros((N, 1), dtype=complex)
    start[N // 2, 0] = 1.0

    vec = np.kron(coin_hilbert(symmetrical), start)


    rho = vec @ vec.conj().T

    P0_coin = np.array([[1, 0],
                        [0, 0]], dtype=complex)

    P1_coin = np.array([[0, 0],
                        [0, 1]], dtype=complex)

    P0 = np.kron(P0_coin, I)
    P1 = np.kron(P1_coin, I)

    data = []

    for _ in range(steps):


        rho = U @ rho @ U.conj().T

        if np.random.rand() <= meas_odds:
            rho = P0 @ rho @ P0 + P1 @ rho @ P1
        
        prob_pos = np.zeros(N)

        for x in range(N):
            prob_pos[x] = (
                rho[x, x].real +
                rho[N + x, N + x].real
            )

        data.append(prob_pos)

    return data, np.array(range(-N//2, N//2))
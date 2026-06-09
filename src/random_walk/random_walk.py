import numpy as np


def _build_transition_matrix_rw_1d(
        walk_size: int, 
        p: float, 
        handle_corners: str | None
    ):
    
    trans_matrix = np.zeros([walk_size, walk_size])

    for i in range(1, walk_size - 1):
        if i + 1 < walk_size:
            trans_matrix[i][(i + 1)] = p
        if i - 1 > -1:
            trans_matrix[i][(i - 1)] = 1 - p

    match handle_corners:
        case "absorbing":
            trans_matrix[0][0] = 1
            trans_matrix[walk_size - 1][walk_size - 1] = 1

        case "reflecting":
            trans_matrix[0][0] = 1 - p
            trans_matrix[0][1] = p
            trans_matrix[walk_size - 1][walk_size - 2] = 1 - p
            trans_matrix[walk_size - 1][walk_size - 1] = p

        case "circular":
            trans_matrix[0][walk_size - 1] = 1 - p
            trans_matrix[0][1] = p
            trans_matrix[walk_size - 1][walk_size - 2] = 1 - p
            trans_matrix[walk_size - 1][0] = p
        
        case _:
            trans_matrix[0][1] = p
            trans_matrix[walk_size - 1][walk_size - 2] = 1- p

    return trans_matrix


def random_walk(
        max_pos: int, 
        steps: int, 
        p: float, 
        handle_corners: str = None
    ):

    walk_size = max_pos * 2 + 1
    arr = np.zeros(walk_size)
    arr[max_pos] = 1 
    trans_matrix = _build_transition_matrix_rw_1d(
        walk_size, 
        p, 
        handle_corners
    )

    r = arr
    for _ in range(steps):
        r = r @ trans_matrix
    return r, list(range(-max_pos, +max_pos + 1))


def random_walk_simulation(
        steps: int, 
        p: float
    ):

    coin = np.random.rand(steps)
    res = [1 if c >= 1 - p else -1 for c in coin]
    path = np.cumsum(res)
    arrival = path[-1]
    return path, arrival

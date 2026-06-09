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
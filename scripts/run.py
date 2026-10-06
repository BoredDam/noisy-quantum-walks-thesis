import matplotlib.pyplot as plt
from src.quantum_random_walk.optimized_quantum_random_walk import circle_quantum_random_walk_1D, noisy_meas_circle_quantum_random_walk_1D, noisy_circle_quantum_random_walk_1D
from src.random_walk.random_walk import random_walk
import numpy as np
import csv
import os

# PARAMETERS

SEED = 42                       # seed for the random number generation
start = 10                    # smallest amount of steps of the QRWs
stop = 100                     # biggest amount of steps of the QRWs
skip = 10                       # how big is the skip in the step count
IS_SYMMETRICAL = True           # must the QRW be symmetrical?
noise_type = 'meas'             # noise type, in ['X', 'Z', 'H', 'meas']
SAMPLE_COUNT = 50               # how many samples for each experiment (higher TIMES, more accurate values)
PLOT = True                    # at the end of the simulation, plot the computed data



max_pos = stop + 1

def var_mu0(X, Y):
    '''
    computes variance of a distribution if mean is equal to 0
    '''
    X_arr = np.asarray(X)
    Y_arr = np.asarray(Y)
    return np.sum(Y_arr * (X_arr ** 2))



# creates the directory

directory_name = f"./data/diffusion_alpha_steps_({start},{stop},{skip})"
try:
    os.mkdir(directory_name)
    print(f"directory '{directory_name}' created successfully.")
except PermissionError:
    print(f"permission denied: unable to create '{directory_name}'.")
    exit(1)
except FileExistsError:
    print(f"directory '{directory_name}' already exists.")
except Exception as e:
    print(f"an error occurred: {e}")
    exit(1)



# actual simulation

ALPHAS = []
P = []

np.random.seed(SEED)

for p in np.linspace(0, 1, 75):

    p = round(p, 2)
    var_ft = []
    t = []
    sum_data = None

    for count in range(SAMPLE_COUNT):
        print(count)
        if noise_type not in ['X', 'Z', 'H', 'meas']:
            print(f"{noise_type} is not a valid noise type.")
            exit(1)

        if noise_type in ['X', 'Z', 'H']:
            data, X = noisy_circle_quantum_random_walk_1D(
                max_pos,
                stop,
                0.5,
                IS_SYMMETRICAL,
                noise_type,
                p,
                None
            )

        if noise_type == 'meas':
            data, X = noisy_meas_circle_quantum_random_walk_1D(
                max_pos,
                stop,
                0.5,
                IS_SYMMETRICAL,
                p,
                None
            )

        data = np.asarray(data, dtype=float)

        if sum_data is None:
            sum_data = np.zeros_like(data)

        sum_data += data

    sum_data /= sum_data.sum(axis=1, keepdims=True)

    for i in range(start, stop + 1, skip):

        t.append(i + 1)

        Y = sum_data[i]

        var_ft.append(var_mu0(X, Y))


    log_t = np.log(t)
    log_var_ft = np.log(var_ft)

    alpha, log_C = np.polyfit(
        log_t,
        log_var_ft,
        1
    )

    print(f'p noise: {p:.3f}, alpha: {alpha:.4f}')

    ALPHAS.append(alpha)
    P.append(p)



with open(f"{directory_name}/{noise_type}.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(P)
    csvwriter.writerow(ALPHAS)

if PLOT:
    plt.plot(P, ALPHAS)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\alpha$")
    plt.grid()
    plt.show()

print("data computed with success.")
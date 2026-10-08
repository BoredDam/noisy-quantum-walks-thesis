import matplotlib.pyplot as plt
from src.quantum_random_walk.optimized_quantum_random_walk import circle_quantum_random_walk_1D, noisy_meas_circle_quantum_random_walk_1D, noisy_circle_quantum_random_walk_1D
from src.random_walk.random_walk import random_walk
import numpy as np
import csv
import os

# PARAMETERS

SEED = 42                       # seed for the random number generation
start = 10                      # smallest amount of steps of the QRWs
stop = 1000                     # biggest amount of steps of the QRWs
skip = 10                       # how big is the skip in the step count
p_amount = 100                  # count of p tested in [0, 1]
IS_SYMMETRICAL = True           # must the QRW be symmetrical?
noise_type = 'X'                # noise type, in ['X', 'Z', 'H', 'meas']
SAMPLE_COUNT = 300              # how many samples for each experiment (higher TIMES, more accurate values)
PLOT = True                     # at the end of the simulation, plot the computed data



MAX_POS = stop + 1

def var_mu0(X, Y):
    '''
    computes variance of a distribution if mean is equal to 0
    '''
    X_arr = np.asarray(X)
    Y_arr = np.asarray(Y)
    return np.sum(Y_arr * (X_arr ** 2))

# 
print(f'''
this is a noisy quantum random walk monte carlo simulation, used to
evaluate alpha, the diffusion exponent that describes the regime 
of a diffusive process, in the context of a noisy channel. 
alpha will be evalued in respect to p, the noise probability 
associated to the quantum channel.

noise type   = {noise_type}
sample count = {SAMPLE_COUNT}
p amount     = {p_amount}
seed         = {SEED}
start        = {start}
stop         = {stop}
skip         = {skip}
symmetrical  = {IS_SYMMETRICAL}
plot         = {PLOT}''')

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

for p in np.linspace(0, 1, p_amount):

    p = round(p, 2)
    var_ft = []
    t = []
    sum_data = None

    for count in range(SAMPLE_COUNT):
        #print(count)
        if noise_type not in ['X', 'Z', 'H', 'meas']:
            print(f"{noise_type} is not a valid noise type.")
            exit(1)

        if noise_type in ['X', 'Z', 'H']:
            data, X = noisy_circle_quantum_random_walk_1D(
                MAX_POS,
                stop,
                0.5,
                IS_SYMMETRICAL,
                noise_type,
                p,
                None
            )

        if noise_type == 'meas':
            data, X = noisy_meas_circle_quantum_random_walk_1D(
                MAX_POS,
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
    noise_title_dict = {
        'meas' : 'generalized dephasing',
        'H'    : 'Hadamard',
        'X'    : 'X-dephasing (bit-flip)',
        'Z'    : 'Z-dephasing (phase-flip)'
    }
    plt.plot(P, ALPHAS)
    plt.title(f'QRW diffusion exponent in a noisy channel,\nnoise type: {noise_title_dict[noise_type]}')
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\alpha$")
    plt.grid()
    plt.show()

print("data computed with success.")
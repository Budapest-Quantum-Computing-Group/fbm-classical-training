import numpy as np
from qml_benchmarks.models.energy_based_model import RestrictedBoltzmannMachine
from tqdm import tqdm

from src.utils import median_heuristic
from src.classical_utils import biased_sample_mmd, get_sample_covariance_matrix

"""
Trainign of Restricted Boltzmann machines.
Includes tuning of the number of hidden unites.
"""

PATH = "./data/molecular/"

CONFIG = dict(
    n_reps = 5, # number of repetitions to average over
    test_length_cutoff = 5, # maximal Z-string length
    hidden_dim = None, # this allows tuning of the hidden unit number
    num_samples = 10000, # number of test samples for tuning
    learning_rate = 0.01, # learning rate for tuning and final training
    ITER = 1000, # number of epochs for tuning and training
    num_samples_training = 50000 # number of test samples for final training
)

def _fine_tune_hidden_dim(
    hidden_dims, learning_rate, ITER, data, test_data, sigmas, num_samples, n_reps
):
    inv_scores = []
    for h in tqdm(hidden_dims):
        model = RestrictedBoltzmannMachine(
            h, verbose=0, learning_rate=learning_rate, n_iter=ITER
        )
        model.fit(data)
        inv_score = 0
        for _ in range(n_reps):
            samples = model.sample(num_samples)
            mmds = biased_sample_mmd(test_data, samples, sigmas)
            inv_score += np.sum(mmds) / n_reps
        inv_scores += [inv_score]

    return inv_scores


def main():
    data = np.loadtxt(PATH + "training_set.txt")
    test_data = np.loadtxt(PATH + "test_set.txt")

    n_reps = CONFIG["n_reps"]
    test_length_cutoff = CONFIG["test_length_cutoff"]
    tune_rbm = CONFIG["tune_rbm"]
    num_samples = CONFIG["num_samples"]
    learning_rate = CONFIG["learning_rate"]
    ITER = CONFIG["ITER"]
    num_samples_training = CONFIG["num_samples_training"]

    n_bits = len(test_data[0])
    sigma = np.sqrt(median_heuristic(data))
    sigmas = np.arange(sigma, sigma**2, 0.5)

    if hidden_dim is None:
        hidden_dims = list(range(n_bits // 2, n_bits, 2))

        inv_scores = _fine_tune_hidden_dim(
            hidden_dims,
            learning_rate,
            ITER,
            data,
            test_data,
            sigmas,
            num_samples,
            n_reps,
        )
        np.savetxt(PATH + "rbm_inv_scores.txt", inv_scores)
        hidden_dim = hidden_dims[np.argmin(inv_scores)]

    model = RestrictedBoltzmannMachine(
        hidden_dim, verbose=0, learning_rate=learning_rate, n_iter=ITER
    )
    model.fit(data)
    mmds = []
    for n in range(n_reps):
        samples = model.sample(num_samples_training)
        mmds.append(biased_sample_mmd(test_data, samples, sigmas, test_length_cutoff))

    np.savetxt(PATH + "rbm_mmds.txt", mmds)

    covariance_matrix = get_sample_covariance_matrix(n_bits // 3, samples)
    np.savetxt(PATH + "rbm_covariance_matrix.txt", covariance_matrix)


if __name__ == "__main__":
    main()

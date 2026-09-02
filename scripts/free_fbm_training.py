import jax.numpy as jnp

import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import (
    median_heuristic,
    get_covariance_matrix,
    test_model,
    get_loss_and_grad_estimator,
    set_seed,
)


PATH = "./data/molecular/"

CONFIG = dict(
    N=20,  # Number of 4-qubit subsystems
    n_layers=7,  # Number of FLO layers
    no_of_Z_samples=1000,  # Number of Z-string samples for MMD estimation
    learning_rate=0.01,  # Learning rate for the optimizer
    ITER=20,  # Number of training iterations
    SEED=None,  # Random seed for reproducibility
    length_cutoff=2,  # Cutoff for Z-string-lengths
    # Physical indices in range(4 * N). None preserves the standard encoding.
    discarded_qubits=None,
)

TEST_CONFIG = dict(
    test_length_cutoff=2,
    n_test_reps=5,
)
# TEST_CONFIG = None  # Disable testing


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])

    N = CONFIG["N"]
    n_layers = CONFIG["n_layers"]
    no_of_Z_samples = CONFIG["no_of_Z_samples"]
    learning_rate = CONFIG["learning_rate"]
    ITER = CONFIG["ITER"]
    length_cutoff = CONFIG["length_cutoff"]
    discarded_qubits = CONFIG["discarded_qubits"]

    n_bits = 4 * N

    d = 2 * n_bits
    no_of_weights_per_layer = d * (d - 1) // 2
    no_of_weights = n_layers * no_of_weights_per_layer

    training_set = np.loadtxt(PATH + "training_set.txt")
    training_set_size = len(training_set)
    heuristic_sigma = np.sqrt(median_heuristic(training_set))
    print("Heuristic sigma: ", heuristic_sigma)

    sigmas = [heuristic_sigma, 2 * heuristic_sigma]

    weights = jnp.zeros(shape=(no_of_weights,)) + jnp.array(
        np.random.rand(no_of_weights)
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    estimate_loss_and_grad = get_loss_and_grad_estimator(
        sigmas=sigmas,
        no_of_Z_samples=no_of_Z_samples,
        N=N,
        training_set=training_set,
        length_cutoff=length_cutoff,
        mode="free_fbm",
        discarded_qubits=discarded_qubits,
    )

    losses = []
    TVs = []
    for i in tqdm(range(ITER)):
        loss, loss_grad = estimate_loss_and_grad(weights)
        losses.append(loss)
        updates, opt_state = optimizer.update(loss_grad, opt_state)
        weights = optax.apply_updates(weights, updates)

    np.savetxt(PATH + "cl_fermion_loss.txt", losses)
    np.savetxt(PATH + "cl_fermion_weights.txt", weights)

    if TEST_CONFIG is None:
        exit()

    print("Testing trained model...")

    test_sigmas = np.arange(heuristic_sigma, heuristic_sigma**2, 0.5)
    test_set = np.loadtxt(PATH + "test_set.txt")

    test_length_cutoff = TEST_CONFIG["test_length_cutoff"]
    n_test_reps = TEST_CONFIG["n_test_reps"]

    test_mmd_mean, test_mmd_std = test_model(
        weights,
        test_sigmas,
        test_length_cutoff,
        n_test_reps,
        test_set,
        N,
        no_of_Z_samples,
        mode="free_fbm",
        discarded_qubits=discarded_qubits,
    )

    np.savetxt(PATH + "cl_fermion_mmds.txt", [test_mmd_mean, test_mmd_std])

    covariance_matrix = get_covariance_matrix(
        N, weights, discarded_qubits=discarded_qubits
    )

    np.savetxt(PATH + "cl_fermion_covariance.txt", covariance_matrix)
    plt.imshow(covariance_matrix)
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(losses)
    axs[0].set_yscale("log")
    axs[1].errorbar(
        test_sigmas,
        test_mmd_mean,
        yerr=test_mmd_std,
        marker="o",
        linestyle="dashed",
    )
    plt.show()

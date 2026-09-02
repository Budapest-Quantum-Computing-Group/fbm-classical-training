import jax.numpy as jnp

import optax
import numpy as np
from tqdm import tqdm

from src import get_loss_and_grad_estimator_exact, get_exact_prob_dist, set_seed


PATH = "./data/mrf/0/"

CONFIG = dict(
    sigma=1.0,  # Gaussian kernel variance
    N=4,  # Number of 4-qubit subsystems
    length_cutoff=5,  # Cutoff for Z-string-lengths
    learning_rate=0.01,  # Learning rate for the optimizer
    ITER=1,  # Number of training iterations
    SEED=None,  # Random seed for reproducibility
    # Physical indices in range(4 * N). None preserves the standard encoding.
    discarded_qubits=None,
)


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])

    sigma = CONFIG["sigma"]
    N = CONFIG["N"]
    length_cutoff = CONFIG["length_cutoff"]
    learning_rate = CONFIG["learning_rate"]
    ITER = CONFIG["ITER"]
    discarded_qubits = CONFIG["discarded_qubits"]

    n_bits = 4 * N
    d = 2 * n_bits
    no_of_weights = d * (d - 1) // 2

    training_set = np.loadtxt(PATH + "training_set.txt")
    target_dist = np.loadtxt(PATH + "target_dist.txt")
    training_set_size = len(training_set)

    alphas = jnp.zeros(shape=(N,)) + 1.0 * jnp.array(np.random.rand(N))
    thetas = jnp.zeros(shape=(no_of_weights,)) + jnp.array(
        np.random.rand(no_of_weights)
    )
    weights = jnp.concatenate([alphas, thetas])

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    estimate_loss_and_grad = get_loss_and_grad_estimator_exact(
        sigma=sigma,
        N=N,
        training_set=training_set,
        length_cutoff=length_cutoff,
        discarded_qubits=discarded_qubits,
    )

    losses = []
    TVs = []

    for i in tqdm(range(ITER)):
        loss, loss_grad = estimate_loss_and_grad(weights)
        losses.append(loss)

        pred_dist = get_exact_prob_dist(
            N, weights[:N], weights[N:], discarded_qubits=discarded_qubits
        )
        TVs.append(np.sum(np.abs(pred_dist - target_dist)) / 2)

        updates, opt_state = optimizer.update(loss_grad, opt_state)
        weights = optax.apply_updates(weights, updates)

        losses.append(loss)

    np.savetxt(PATH + "fermion_loss_" + str(length_cutoff) + ".txt", losses)
    np.savetxt(PATH + "fermion_tvd_" + str(length_cutoff) + ".txt", TVs)

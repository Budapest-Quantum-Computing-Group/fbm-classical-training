import numpy as np

import numba as nb

import jax.numpy as jnp

from scipy.special import comb
from src.piquasso_utils import get_fock_subspace_dimension


def get_visible_qubits(N, discarded_qubits=None):
    """Return physical FBM qubits retained for data observables.

    By default, the first qubit of each four-qubit subsystem is discarded,
    preserving the original FBM encoding. Passing ``discarded_qubits`` replaces
    that default with explicit indices from the full ``4 * N``-qubit circuit.
    """
    n_physical_qubits = 4 * N
    if discarded_qubits is None:
        discarded_qubits = range(0, n_physical_qubits, 4)

    discarded_qubits = tuple(discarded_qubits)
    if any(
        not isinstance(qubit, (int, np.integer)) or isinstance(qubit, bool)
        for qubit in discarded_qubits
    ):
        raise TypeError("discarded_qubits must contain only integer indices.")
    if len(set(discarded_qubits)) != len(discarded_qubits):
        raise ValueError("discarded_qubits must not contain duplicate indices.")
    if any(qubit < 0 or qubit >= n_physical_qubits for qubit in discarded_qubits):
        raise ValueError(
            f"discarded_qubits must contain indices from 0 to {n_physical_qubits - 1}."
        )
    if len(discarded_qubits) >= n_physical_qubits:
        raise ValueError("At least one physical qubit must be retained.")

    return np.delete(np.arange(n_physical_qubits), discarded_qubits)


def get_num_visible_qubits(N, discarded_qubits=None):
    """Return the number of physical FBM qubits retained for data observables."""
    return len(get_visible_qubits(N, discarded_qubits))


def get_localities_frequencies(
    sigma,
    N,
    length_cutoff,
    no_of_Z_samples,
    discarded_qubits=None,
):
    n_visible_qubits = get_num_visible_qubits(N, discarded_qubits)
    if not 1 <= length_cutoff <= n_visible_qubits:
        raise ValueError(
            f"length_cutoff must be between 1 and {n_visible_qubits}, "
            f"got {length_cutoff}."
        )
    loc_probs = normalize_prob_dist(
        get_weight(
            np.arange(length_cutoff + 1), sigma=sigma, n_bits=n_visible_qubits
        )[1:]
    )
    loc_probs /= np.sum(loc_probs)
    locality_frequencies = np.int_(np.rint(loc_probs * no_of_Z_samples))
    localities = np.arange(1, length_cutoff + 1)

    return localities, locality_frequencies


def median_heuristic(dataset):
    distances = []
    if len(dataset) > 1000:
        idx = np.random.randint(len(dataset), size=1000)
        dataset = dataset[idx]
    for x in dataset:
        for y in dataset:
            distances.append(np.sum(np.abs(x - y)))

    return np.median(distances)


def normalize_prob_dist(array):
    return array / np.sum(array)


@nb.njit(cache=True)
def get_vector_lengths(N, l):
    lengths = np.empty(shape=(l + 1), dtype=nb.int64)
    for l1 in range(l + 1):
        lengths[l1] = get_fock_subspace_dimension(N, l1)

    return lengths


def get_weight(x, sigma, n_bits):
    p_sigma = (1 - np.exp(-1 / (2 * sigma))) / 2

    return comb(n_bits, x) * (1 - p_sigma) ** (n_bits - x) * p_sigma**x


def get_weight_vector(n_bits, loc_probs):
    vec = []
    for i in range(1, len(loc_probs) + 1):
        vec.append(loc_probs[i - 1] * np.ones(int(comb(n_bits, i))))

    return jnp.concatenate(vec)

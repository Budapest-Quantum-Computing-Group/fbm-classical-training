import numpy as np

import numba as nb

import jax.numpy as jnp

from piquasso._math.combinatorics import comb


def get_localities_frequencies(sigma, N, length_cutoff, no_of_Z_samples):
    loc_probs = normalize_prob_dist(
        get_weight(np.arange(length_cutoff + 1), sigma=sigma, n_bits=(3 * N))[1:]
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
        lengths[l1] = comb(N, l1)

    return lengths


def get_weight(x, sigma, n_bits):
    p_sigma = (1 - np.exp(-1 / (2 * sigma))) / 2

    from scipy.special import comb

    return comb(n_bits, x) * (1 - p_sigma) ** (n_bits - x) * p_sigma**x


def get_weight_vector(n_bits, loc_probs):
    vec = []
    for i in range(1, len(loc_probs) + 1):
        vec.append(loc_probs[i - 1] * np.ones(int(comb(n_bits, i))))

    return jnp.concatenate(vec)

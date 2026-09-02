import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src.gaussian_decomposition import (
    calculate_Z_string_expvals,
    calculate_Z_string_expvals_free,
)
from src.utils import get_localities_frequencies, get_visible_qubits
from src.indices import (
    expand_to_Z,
    map_visible_to_physical,
    random_first_quantized_majorana_string,
)
from src.encoding import get_target_Z_string

from typing import Literal


def get_covariance_matrix(N, weights, discarded_qubits=None):
    """Calculate covariance on retained visible qubits.

    ``discarded_qubits`` contains physical indices in the full ``4 * N``-qubit
    circuit. The returned axes follow the retained physical-qubit order.
    """
    visible_qubits = get_visible_qubits(N, discarded_qubits)
    n_visible_qubits = len(visible_qubits)
    single_Z_strings = []
    double_Z_strings = []
    for i in range(n_visible_qubits):
        Z_string = np.array([i])
        majorana_string = expand_to_Z(
            map_visible_to_physical(Z_string, visible_qubits)
        )
        single_Z_strings.append(majorana_string)
        for j in range(i + 1, n_visible_qubits):
            Z_string = np.array([i, j])
            majorana_string = expand_to_Z(
                map_visible_to_physical(Z_string, visible_qubits)
            )
            double_Z_strings.append(majorana_string)

    all_Z_strings = [
        np.empty((0, 0)),
        np.array(single_Z_strings),
        np.array(double_Z_strings),
    ]

    Z_string_expvals = calculate_Z_string_expvals(weights, all_Z_strings, N)
    Z_expvals = Z_string_expvals[:n_visible_qubits]
    ZZ_expvals = Z_string_expvals[n_visible_qubits:]
    cov_matrix = np.ones(shape=(n_visible_qubits, n_visible_qubits))
    indices = np.triu_indices(n_visible_qubits, k=1)
    cov_matrix[indices[0], indices[1]] = ZZ_expvals
    cov_matrix[indices[1], indices[0]] = ZZ_expvals

    cov_matrix -= np.outer(Z_expvals, Z_expvals)

    return cov_matrix


def test_model(
    weights,
    sigmas,
    length_cutoff,
    n_repetitions,
    test_set,
    N,
    no_of_Z_samples,
    mode: Literal["fbm", "free_fbm"] = "fbm",
    discarded_qubits=None,
):
    """Estimate test MMD on the retained visible qubits."""
    if mode == "fbm":
        calculate_Z_string_expvals_func = calculate_Z_string_expvals
    elif mode == "free_fbm":
        calculate_Z_string_expvals_func = calculate_Z_string_expvals_free
    else:
        raise NotImplemented(f"Mode {mode} not implemented.")

    visible_qubits = get_visible_qubits(N, discarded_qubits)
    n_visible_qubits = len(visible_qubits)
    test_mmds = []
    for i in tqdm(range(n_repetitions)):
        for sigma in sigmas:
            localities, locality_frequencies = get_localities_frequencies(
                sigma=sigma,
                N=N,
                length_cutoff=length_cutoff,
                no_of_Z_samples=no_of_Z_samples,
                discarded_qubits=discarded_qubits,
            )
            sampled_Z_strings = [np.empty((0, 0), dtype=int)]
            target_Z_expvals = [np.empty((0, 0), dtype=int)]
            for locality, freq in zip(localities, locality_frequencies):
                sampled_Z_strings_local = np.empty((freq, 2 * locality), dtype=int)
                target_Z_expvals_local = np.empty(freq, dtype=weights.dtype)
                for i in range(freq):
                    sampled_majorana_string = random_first_quantized_majorana_string(
                        n_visible_qubits, locality
                    )
                    physical_string = map_visible_to_physical(
                        sampled_majorana_string, visible_qubits
                    )
                    sampled_Z_string = expand_to_Z(physical_string)
                    sampled_Z_strings_local[i] = sampled_Z_string
                    target_Z_expvals_local[i] = get_target_Z_string(
                        test_set, sampled_majorana_string
                    )
                sampled_Z_strings.append(sampled_Z_strings_local)
                target_Z_expvals.append(target_Z_expvals_local)

            Z_string_expvals = calculate_Z_string_expvals_func(
                weights, sampled_Z_strings, N
            )
            test_mmds.append(
                jnp.mean(
                    (jnp.concatenate(target_Z_expvals[1:]) - Z_string_expvals) ** 2
                )
            )

    test_mmds = np.array(test_mmds).reshape((n_repetitions, len(sigmas)))
    test_mmd_mean = np.mean(test_mmds, axis=0)
    test_mmd_std = np.std(test_mmds, axis=0)

    return test_mmd_mean, test_mmd_std

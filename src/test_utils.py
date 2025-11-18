import jax
import jax.numpy as jnp
import numpy as np
from piquasso.fermionic._utils import fock_to_binary_indices
import piquasso as pq
from tqdm import tqdm

from src.exact import matchgate_circuit
from src.indices import (
    pad_majorana_string,
    expand_to_Z,
)
from src.gaussian_decomposition import (
    calculate_Z_string_expvals,
    calculate_Z_string_expvals_free,
)
from src.utils import get_localities_frequencies
from src.indices import (
    pad_majorana_string,
    expand_to_Z,
    random_first_quantized_majorana_string,
)
from src.encoding import get_target_Z_string

from typing import Literal


def get_exact_prob_dist(N, alphas, thetas):
    zeros = [0, 0, 0, 0]
    ones = [1, 1, 1, 1]
    occupation_numbers = [zeros, ones]
    funcs = [np.cos, np.sin]

    n_bits = 4 * N

    with pq.Program() as preparation:
        from piquasso.fermionic._utils import get_fock_space_basis

        for binary in get_fock_space_basis(N, N + 1):
            pq.Q() | pq.StateVector(
                sum([occupation_numbers[i] for i in binary], [])
            ) * np.prod(
                [funcs[i](alphas[idx]) for idx, i in enumerate(binary.tolist())]
            )

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.Program(matchgate_circuit(thetas))

    simulator = pq.fermionic.PureFockSimulator(
        d=n_bits, config=pq.Config(cutoff=n_bits + 1, validate=False)
    )

    state = simulator.execute(program).state
    pred_dist = state.get_marginal_fock_probabilities_on_modes(
        np.delete(np.arange(n_bits), np.arange(0, n_bits, 4))
    )

    return pred_dist[fock_to_binary_indices(n_bits - N)]


def get_covariance_matrix(N, weights):
    single_Z_strings = []
    double_Z_strings = []
    for i in range(3 * N):
        Z_string = np.array([i])
        majorana_string = expand_to_Z(pad_majorana_string(Z_string, 4 - 1))
        single_Z_strings.append(majorana_string)
        for j in range(i + 1, 3 * N):
            Z_string = np.array([i, j])
            majorana_string = expand_to_Z(pad_majorana_string(Z_string, 4 - 1))
            double_Z_strings.append(majorana_string)

    all_Z_strings = [
        np.empty((0, 0)),
        np.array(single_Z_strings),
        np.array(double_Z_strings),
    ]

    Z_string_expvals = calculate_Z_string_expvals(weights, all_Z_strings, N)
    Z_expvals = Z_string_expvals[: 3 * N]
    ZZ_expvals = Z_string_expvals[3 * N :]
    cov_matrix = np.ones(shape=(3 * N, 3 * N))
    indices = np.triu_indices(3 * N, k=1)
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
):
    if mode == "fbm":
        calculate_Z_string_expvals_func = calculate_Z_string_expvals
    elif mode == "free_fbm":
        calculate_Z_string_expvals_func = calculate_Z_string_expvals_free
    else:
        raise NotImplemented(f"Mode {mode} not implemented.")

    test_mmds = []
    for i in tqdm(range(n_repetitions)):
        for sigma in sigmas:
            localities, locality_frequencies = get_localities_frequencies(
                sigma=sigma, N=N, length_cutoff=length_cutoff, no_of_Z_samples=no_of_Z_samples
            )
            sampled_Z_strings = [np.empty((0, 0), dtype=int)]
            target_Z_expvals = [np.empty((0, 0), dtype=int)]
            for locality, freq in zip(localities, locality_frequencies):
                sampled_Z_strings_local = np.empty((freq, 2 * locality), dtype=int)
                target_Z_expvals_local = np.empty(freq, dtype=weights.dtype)
                for i in range(freq):
                    sampled_majorana_string = random_first_quantized_majorana_string(
                        3 * N, locality
                    )
                    sampled_Z_string = expand_to_Z(
                        pad_majorana_string(sampled_majorana_string, 4 - 1)
                    )
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

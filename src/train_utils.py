import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from src.indices import calculate_relevant_Z_strings_on_subspace
from src.encoding import get_target_Z_strings_from_samples, get_target_Z_string
from src.gaussian_decomposition import (
    calculate_Z_string_expvals,
    calculate_Z_string_expvals_free,
)
from src.utils import (
    get_weight,
    normalize_prob_dist,
    get_weight_vector,
    get_localities_frequencies,
    get_num_visible_qubits,
    get_visible_qubits,
)
from src.indices import (
    expand_to_Z,
    map_visible_to_physical,
    random_first_quantized_majorana_string,
)

from typing import Literal


def _calculate_loss_weighted(weights, Z_strings, target_expvals, N, loc_weights):
    Z_string_expvals = calculate_Z_string_expvals(weights, Z_strings, N)[1:]

    return np.sum(loc_weights * (target_expvals - Z_string_expvals) ** 2) / np.sum(
        loc_weights
    )


def get_loss_and_grad_estimator_exact(
    sigma,
    N,
    training_set,
    length_cutoff,
    discarded_qubits=None,
):
    """
    Args:
        sigma: Gaussian kernel variance.
        N: Number of subsystems.
        training_set: For calculating target_Z_strings on-demand.
        length_cutoff: Maximum locality for Z strings.
        discarded_qubits: Physical qubit indices to exclude. By default, the
            first qubit of every four-qubit subsystem is excluded.
    """
    n_bits = 4 * N
    n_visible_qubits = get_num_visible_qubits(N, discarded_qubits)

    target_Z_strings: list = get_target_Z_strings_from_samples(
        training_set, n_visible_qubits, length_cutoff
    )
    target_Z_strings = jnp.concatenate(target_Z_strings)

    Z_strings = [
        calculate_relevant_Z_strings_on_subspace(
            n_bits, length, N, discarded_qubits
        )
        for length in range(length_cutoff + 1)
    ]

    loc_probs = normalize_prob_dist(
        get_weight(
            np.arange(length_cutoff + 1), sigma=sigma, n_bits=n_visible_qubits
        )[1:]
    )
    loc_weights = get_weight_vector(n_visible_qubits, loc_probs)

    def _estimate_loss_and_grad(weights):
        partial_calculate_loss = partial(
            _calculate_loss_weighted,
            Z_strings=Z_strings,
            target_expvals=target_Z_strings,
            N=N,
            loc_weights=loc_weights,
        )

        return jax.value_and_grad(partial_calculate_loss)(weights)

    return _estimate_loss_and_grad


def _calculate_loss(weights, Z_strings, target_expvals, N):
    Z_string_expvals = calculate_Z_string_expvals(weights, Z_strings, N)

    return jnp.mean((jnp.concatenate(target_expvals[1:]) - Z_string_expvals) ** 2)


def _calculate_loss_gauss(weights, Z_strings, target_expvals, N):
    Z_string_expvals = calculate_Z_string_expvals_free(weights, Z_strings, N)

    return jnp.mean((jnp.concatenate(target_expvals[1:]) - Z_string_expvals) ** 2)


def get_loss_and_grad_estimator(
    sigmas,
    no_of_Z_samples,
    N,
    training_set,  # For calculating target_Z_strings on-demand.
    length_cutoff,
    mode: Literal["fbm", "free_fbm"] = "fbm",
    discarded_qubits=None,
):
    """Returns a function that estimates the loss and its gradient.

    Args:
        sigmas (list): List of Gaussian kernel variances.
        no_of_Z_samples (int): Number of Z samples.
        N (int): Number of subsystems.
        training_set (array-like): Training data for calculating target Z strings on-demand.
        length_cutoff (int): Maximum locality for Z strings.
        mode (Literal["fbm", "free_fbm"], optional): Mode of operation. Defaults to "fbm".
        discarded_qubits (collection[int], optional): Physical qubit indices to
            exclude from the loss. Defaults to the first qubit of every
            four-qubit subsystem.
    """

    if mode == "fbm":
        loss_calculation_fn = _calculate_loss
    elif mode == "free_fbm":
        loss_calculation_fn = _calculate_loss_gauss
    else:
        raise NotImplemented(f"Mode {mode} not implemented.")

    visible_qubits = get_visible_qubits(N, discarded_qubits)
    n_visible_qubits = len(visible_qubits)
    locality_all_sigma = []
    locality_frequencies_all_sigma = []
    for sigma in sigmas:
        localities_for_sigma, locality_frequencies_for_sigma = (
            get_localities_frequencies(
                sigma=sigma,
                N=N,
                length_cutoff=length_cutoff,
                no_of_Z_samples=no_of_Z_samples,
                discarded_qubits=discarded_qubits,
            )
        )
        locality_all_sigma.append(localities_for_sigma)
        locality_frequencies_all_sigma.append(locality_frequencies_for_sigma)

    localities = locality_all_sigma[0]
    locality_frequencies = np.sum(locality_frequencies_all_sigma, axis=0)

    def _estimate_loss_and_grad(weights):
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
                    training_set, sampled_majorana_string
                )
            sampled_Z_strings.append(sampled_Z_strings_local)
            target_Z_expvals.append(target_Z_expvals_local)

        partial_calculate_loss = partial(
            loss_calculation_fn,
            Z_strings=sampled_Z_strings,
            target_expvals=target_Z_expvals,
            N=N,
        )

        return jax.value_and_grad(partial_calculate_loss)(weights)

    return _estimate_loss_and_grad

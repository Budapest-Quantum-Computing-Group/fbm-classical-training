"""Local combinatorics and optional Piquasso-backed debugging utilities.

The approximate FBM training path uses the local combinatorial functions in
this module and does not require Piquasso. Exact Fock-space simulation and the
debugging helpers still need Piquasso, which is imported lazily when one of
those functions is called.
"""

from dataclasses import dataclass
from itertools import product

import numba as nb
import numpy as np


@nb.njit(cache=True)
def get_fock_subspace_dimension(d, particle_number):
    """Return ``binomial(d, particle_number)`` using integer arithmetic."""
    if particle_number < 0 or particle_number > d:
        return 0

    k = min(particle_number, d - particle_number)
    result = 1
    for i in range(1, k + 1):
        result = result * (d - k + i) // i
    return result


@nb.njit(cache=True)
def next_first_quantized(string, d):
    """Return the next lexicographic fixed-particle-number index vector."""
    result = string.copy()
    locality = len(result)
    for i in range(locality - 1, -1, -1):
        if result[i] < d - locality + i:
            result[i] += 1
            for j in range(i + 1, locality):
                result[j] = result[j - 1] + 1
            return result
    return result


@nb.njit(cache=True)
def to_first_quantized(bitstring):
    """Convert a binary occupation vector to occupied-mode indices."""
    result = np.empty(np.sum(bitstring), dtype=np.int64)
    result_index = 0
    for index in range(len(bitstring)):
        if bitstring[index]:
            result[result_index] = index
            result_index += 1
    return result


def _require_piquasso():
    try:
        import piquasso as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This exact/debugging utility requires the optional 'piquasso' "
            "package. Install Piquasso separately to use it."
        ) from exc
    return pq


def _get_initial_fock_amplitudes(N, alphas):
    """Return the FBM product-state amplitudes as a Piquasso Fock map."""
    alphas = np.asarray(alphas).reshape(-1)
    if len(alphas) != N:
        raise ValueError(f"Expected {N} preparation angles, got {len(alphas)}.")

    amplitudes = {}
    for binary in product((0, 1), repeat=N):
        occupation = tuple(bit for value in binary for bit in (value,) * 4)
        amplitude = np.prod(
            [
                np.sin(alpha) if value else np.cos(alpha)
                for alpha, value in zip(alphas, binary)
            ]
        )
        amplitudes[occupation] = amplitude
    return amplitudes


@dataclass
class Givens:
    theta: float
    modes: np.ndarray


def _calculate_d_from_len_weights(len_weights):
    return np.int_(np.ceil(np.sqrt(len_weights * 2)))


def matchgate_circuit(weights):
    """Construct the optional Piquasso instruction list for exact simulation."""
    pq = _require_piquasso()
    from src.SO import calculate_modes

    d = _calculate_d_from_len_weights(len(weights))
    if d % 2 != 0:
        raise ValueError("'d' should be even")

    rotations = [
        Givens(theta=weight, modes=modes)
        for weight, modes in zip(weights, np.asarray(calculate_modes(d), dtype=int))
    ]

    instructions = []
    for rotation in rotations:
        mode = rotation.modes[0]
        bit = mode // 2
        if mode % 2 == 0:
            instructions.append(pq.Phaseshifter(rotation.theta).on_modes(bit))
        else:
            instructions.append(
                pq.fermionic.IsingXX(-rotation.theta / 2).on_modes(bit, bit + 1)
            )
    return instructions


def get_exact_prob_dist_piquasso(N, alphas, thetas, discarded_qubits=None):
    """Return the exact distribution on retained visible qubits using Piquasso."""
    pq = _require_piquasso()
    from piquasso.fermionic._utils import (
        fock_to_binary_indices,
        get_fock_space_basis,
    )
    from src.utils import get_visible_qubits

    n_bits = 4 * N

    with pq.Program() as preparation:
        pq.Q() | pq.FockStateVector(_get_initial_fock_amplitudes(N, alphas))

    with pq.Program() as program:
        pq.Q() | preparation
        majorana_dimension = 2 * n_bits
        weights_per_layer = majorana_dimension * (majorana_dimension - 1) // 2
        if len(thetas) % weights_per_layer != 0:
            raise ValueError(
                f"Expected a multiple of {weights_per_layer} FLO weights for N={N}, "
                f"got {len(thetas)}."
            )
        for start in range(0, len(thetas), weights_per_layer):
            layer = thetas[start : start + weights_per_layer]
            pq.Q() | pq.Program(matchgate_circuit(layer))

    simulator = pq.fermionic.PureFockSimulator(
        d=n_bits, config=pq.Config(cutoff=n_bits + 1, validate=False)
    )
    state = simulator.execute(program).state
    visible_modes = get_visible_qubits(N, discarded_qubits)
    if hasattr(state, "get_marginal_fock_probabilities_on_modes"):
        probabilities = state.get_marginal_fock_probabilities_on_modes(visible_modes)
        return probabilities[fock_to_binary_indices(len(visible_modes))]

    # Piquasso 8 removed the marginal helper. Keep this optional reference
    # implementation useful for small debugging circuits by accumulating its
    # Fock-basis probabilities in the repository's binary-index convention.
    basis = get_fock_space_basis(n_bits, n_bits + 1)
    visible_basis = basis[:, visible_modes]
    powers = 1 << np.arange(len(visible_modes) - 1, -1, -1)
    keys = visible_basis @ powers
    probabilities = np.zeros(1 << len(visible_modes), dtype=float)
    np.add.at(probabilities, keys, np.real(state.fock_probabilities))
    return probabilities


def sample_fbm_piquasso(N, alphas, thetas, shots, discarded_qubits=None, seed=None):
    """Sample retained FBM modes using a terminal particle-number measurement."""
    pq = _require_piquasso()
    from src.utils import get_visible_qubits

    n_bits = 4 * N

    with pq.Program() as program:
        pq.Q() | pq.FockStateVector(_get_initial_fock_amplitudes(N, alphas))

        majorana_dimension = 2 * n_bits
        weights_per_layer = majorana_dimension * (majorana_dimension - 1) // 2
        if len(thetas) % weights_per_layer != 0:
            raise ValueError(
                f"Expected a multiple of {weights_per_layer} FLO weights for N={N}, "
                f"got {len(thetas)}."
            )
        for start in range(0, len(thetas), weights_per_layer):
            layer = thetas[start : start + weights_per_layer]
            pq.Q() | pq.Program(matchgate_circuit(layer))

        visible_modes = get_visible_qubits(N, discarded_qubits)
        pq.Q(*map(int, visible_modes)) | pq.ParticleNumberMeasurement()

    simulator = pq.fermionic.PureFockSimulator(
        d=n_bits,
        config=pq.Config(
            cutoff=n_bits + 1,
            seed_sequence=seed,
            validate=False,
        ),
    )
    result = simulator.execute(program, shots=shots)
    return np.asarray(result.samples, dtype=np.int8)


get_exact_prob_dist = get_exact_prob_dist_piquasso
get_fbm_probabilities = get_exact_prob_dist_piquasso
sample_fbm = sample_fbm_piquasso


def get_majorana_operators(d):
    """Return Majorana matrices using Piquasso's optional Fock operators."""
    _require_piquasso()
    from piquasso.fermionic._utils import _get_fs_fdags

    fs, fdags = _get_fs_fdags(d)
    operators = []
    for i in range(d):
        operators.append(fs[i] + fdags[i])
        operators.append(-1j * (fs[i] - fdags[i]))
    return np.array(operators)


def _calculate_covariance_matrix(state_vector, d):
    majoranas = get_majorana_operators(d)
    covariance_matrix = np.empty(shape=(2 * d, 2 * d))
    for i in range(2 * d):
        for j in range(2 * d):
            covariance_matrix[i, j] = np.real(
                -1j
                * (
                    np.vdot(state_vector, majoranas[i] @ majoranas[j] @ state_vector)
                    - (1.0 if i == j else 0.0)
                )
            )
    return covariance_matrix


def get_diagonalizing_unitary_from_covariance_matrix(covariance_matrix):
    """Return the Piquasso Fock-space unitary diagonalizing a covariance matrix."""
    pq = _require_piquasso()
    from piquasso._math.transformations import xxpp_to_xpxp_indices
    from piquasso.fermionic._utils import get_fermionic_hamiltonian, get_omega

    d = len(covariance_matrix) // 2
    connector = pq.NumpyConnector()
    _, orthogonal = connector.schur(covariance_matrix)

    if not np.isclose(np.linalg.det(orthogonal), 1.0):
        flip = connector.block_diag(
            *([np.array([[0, 1], [1, 0]])] + [np.identity(2)] * (d - 1))
        )
        orthogonal = orthogonal @ flip

    omega = get_omega(d)
    basis_transformation = omega[xxpp_to_xpxp_indices(d)]
    i2H = (
        basis_transformation.conj().T
        @ connector.real_logm(orthogonal)
        @ basis_transformation
    )
    bigH = get_fermionic_hamiltonian(i2H / 2j, connector)
    return connector.expm(1j * bigH), orthogonal


def postprocess_state_vector(state_vector, d):
    """Diagonalize a state vector using the optional Piquasso implementation."""
    covariance_matrix = _calculate_covariance_matrix(state_vector, d)
    unitary, orthogonal = get_diagonalizing_unitary_from_covariance_matrix(
        covariance_matrix
    )
    return unitary.conj().T @ state_vector, orthogonal

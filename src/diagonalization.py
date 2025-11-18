from src.majorana_operators import get_majorana_operators
import numpy as np

from piquasso import NumpyConnector

from piquasso.fermionic._utils import get_fermionic_hamiltonian, get_omega

from piquasso._math.transformations import xxpp_to_xpxp_indices


def _calculate_covariance_matrix(state_vector, d):
    ms = get_majorana_operators(d)

    covariance_matrix = np.empty(shape=(2 * d, 2 * d))

    for i in range(2 * d):
        for j in range(2 * d):
            covariance_matrix[i, j] = np.real(
                -1j
                * (
                    np.vdot(state_vector, ms[i] @ ms[j] @ state_vector)
                    - (1.0 if i == j else 0.0)
                )
            )

    return covariance_matrix


def get_diagonalizing_unitary_from_covariance_matrix(covariance_matrix):
    d = len(covariance_matrix) // 2

    connector = NumpyConnector()

    _, O = connector.schur(covariance_matrix)

    is_special_orthogonal: bool = np.isclose(np.linalg.det(O), 1.0)

    if not is_special_orthogonal:
        # NOTE: If the orthogonal coming from the real Schur decomposition is not
        # special, then there is no corresponding real logarithm from which we can
        # calculate the gate hamiltonian. To fix this, we can just perform a flip.
        flip = connector.block_diag(
            *([np.array([[0, 1], [1, 0]])] + [np.identity(2)] * (d - 1))
        )
        O = O @ flip

    # Unitary which diagonalizes the original density matrix
    omega = get_omega(d)

    indices = xxpp_to_xpxp_indices(d)

    basis_transformation = omega[indices]

    i2H = basis_transformation.conj().T @ connector.real_logm(O) @ basis_transformation

    H = i2H / 2j
    bigH = get_fermionic_hamiltonian(H, connector)
    unitary = connector.expm(1j * bigH)

    return unitary, O


def postprocess_state_vector(state_vector, d):
    """
    Postprocess the state vector to get the diagonalizing unitary.
    """
    covariance_matrix = _calculate_covariance_matrix(state_vector, d)

    unitary, O = get_diagonalizing_unitary_from_covariance_matrix(covariance_matrix)

    new_state_vector = unitary.conj().T @ state_vector

    return new_state_vector, O

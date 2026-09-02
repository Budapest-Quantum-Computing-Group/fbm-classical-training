"""Compatibility exports for optional Piquasso diagonalization helpers."""

from src.piquasso_utils import (
    _calculate_covariance_matrix,
    get_diagonalizing_unitary_from_covariance_matrix,
    postprocess_state_vector,
)


__all__ = [
    "_calculate_covariance_matrix",
    "get_diagonalizing_unitary_from_covariance_matrix",
    "postprocess_state_vector",
]

import numpy as np
import pytest

pytest.importorskip("piquasso")

from src.SO import calculate_modes  # noqa: E402
from src.piquasso_utils import (  # noqa: E402
    get_exact_prob_dist,
    sample_fbm,
)


def single_xx_layer(theta):
    weights = np.zeros(28)
    modes = np.asarray(calculate_modes(8))
    first_xx = np.flatnonzero(modes[:, 0] % 2 == 1)[0]
    weights[first_xx] = theta
    return weights


def test_probability_indices_follow_data_column_order():
    theta = 0.73
    probabilities = get_exact_prob_dist(
        1, np.array([0.0]), single_xx_layer(theta), discarded_qubits=()
    )

    expected = np.zeros(16)
    expected[0] = np.cos(theta / 2) ** 2
    expected[12] = np.sin(theta / 2) ** 2  # q0,q1 = 1 -> data word 1100
    np.testing.assert_allclose(probabilities, expected, atol=1e-14)


def test_discarded_qubits_are_marginalized_in_requested_order():
    theta = 0.73
    probabilities = get_exact_prob_dist(
        1, np.array([0.0]), single_xx_layer(theta), discarded_qubits=(0,)
    )

    expected = np.zeros(8)
    expected[0] = np.cos(theta / 2) ** 2
    expected[4] = np.sin(theta / 2) ** 2  # retained word q1,q2,q3 = 100
    np.testing.assert_allclose(probabilities, expected, atol=1e-14)


def test_samples_use_the_same_bit_order():
    samples = sample_fbm(
        1,
        np.array([0.0]),
        single_xx_layer(np.pi),
        shots=10,
        discarded_qubits=(),
        seed=12,
    )

    np.testing.assert_array_equal(samples, np.tile([1, 1, 0, 0], (10, 1)))

import piquasso as pq
import numpy as np

from dataclasses import dataclass

from .SO import calculate_modes


if not __debug__:
    raise ImportError(
        "This module is only for debugging purposes. Please run without the -O flag to "
        "enable debug mode."
    )


@dataclass
class Givens:
    theta: float
    modes: np.array


def _merge(weights, all_modes):
    rotations = []

    for i in range(len(weights)):
        rotations.append(Givens(theta=weights[i], modes=all_modes[i]))

    return rotations


def _calculate_d_from_len_weights(len_weights):
    return np.int_(np.ceil(np.sqrt(len_weights * 2)))


def matchgate_circuit(weights):
    """
    NOTE: This may be compressed by solving Eq. (A22-24).
    """
    d = _calculate_d_from_len_weights(len(weights))

    assert d % 2 == 0, "'d' should be even"

    all_modes = np.array(calculate_modes(d), dtype=int)

    rotations = _merge(weights, all_modes)

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

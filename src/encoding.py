import numpy as np

from src.piquasso_utils import next_first_quantized

from src.utils import get_vector_lengths


def get_target_Z_string(training_set, Z_string):
    # the expval is the mean parity on the corresponding bits
    marginal_data = training_set[:, Z_string]
    parities = (-1) ** np.sum(marginal_data, axis=1)
    return np.mean(parities)


def get_target_Z_strings_from_samples(training_set, n_bits, max_loc):
    sub_vec_lengths = get_vector_lengths(n_bits, max_loc)
    targets = []

    for loc in range(1, max_loc + 1):
        size = sub_vec_lengths[loc]

        target_on_locality = np.zeros(size, dtype=float)
        Z_string = np.arange(loc)
        current_index = 0
        for i in range(size):
            target_on_locality[current_index] = get_target_Z_string(
                training_set, Z_string
            )

            current_index += 1
            Z_string = next_first_quantized(Z_string, n_bits)

        targets.append(target_on_locality)

    return targets

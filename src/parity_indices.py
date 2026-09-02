import numpy as np

from src.indices import iterate_first_quantized_on_fock_subspace

from itertools import product

from src.piquasso_utils import get_fock_subspace_dimension


def ordered_partitions(n):
    return [x + [n - i] for i in range(n) for x in ordered_partitions(i)] or [[]]


def get_all_first_quantized_majorana_strings(max_no_of_pairs, N, d):
    first_quantized_local_majoranas = _get_first_quantized_local_majoranas(
        max_no_of_pairs, d
    )
    majorana_strings = [np.array([[]], dtype=int)]

    for number_of_pairs in range(1, max_no_of_pairs + 1):
        majorana_string_for_pair_number = _get_majorana_strings_for_pair_number(
            first_quantized_local_majoranas, number_of_pairs, N, d
        )

        majorana_strings.append(majorana_string_for_pair_number)

    return majorana_strings


def _get_number_of_possible_majoranas(number_of_pairs, N, d):
    return np.sum(
        [
            _get_ordered_partition_size(op, N, d)
            for op in ordered_partitions(number_of_pairs)
        ]
    )


def _get_majorana_strings(d, number_of_pairs):
    if number_of_pairs == 1:
        return np.arange(0, 2 * d).reshape(d, 2)

    return np.array(
        list(iterate_first_quantized_on_fock_subspace(2 * d, 2 * number_of_pairs))
    )


def _get_first_quantized_local_majoranas(max_no_of_pairs, d):
    """First quantized Majorana strings on a given subsystem with number of qubits `d`.

    Args:
        max_no_of_pairs (int): Maximum number of pairs of Majoranas.
        d (int): Number of qubits (with ancillas, typically 5).
    """
    first_quantized_local_majoranas = []

    for number_of_pairs in range(max_no_of_pairs + 1):
        first_quantized_local_majoranas.append(
            _get_majorana_strings(d, number_of_pairs)
        )

    return first_quantized_local_majoranas


def _get_majorana_strings_for_pair_number(
    first_quantized_local_majoranas, number_of_pairs, N, d
):
    """
    Generate all possible Majorana strings for a given number of pairs.
    """
    i = 0

    size = _get_number_of_possible_majoranas(number_of_pairs, N, d)

    majorana_string_for_pair_number = np.empty(
        shape=(size, 2 * number_of_pairs), dtype=int
    )

    for partition in ordered_partitions(number_of_pairs):
        len_partition = len(partition)

        selected_local_majoranas = [
            first_quantized_local_majoranas[part] for part in partition
        ]

        lengths = [len(x) for x in selected_local_majoranas]

        for arrangement in iterate_first_quantized_on_fock_subspace(N, len_partition):
            for idx in product(*[range(x) for x in lengths]):
                to_concatenate = []

                for kdx, jdx in enumerate(idx):
                    place = arrangement[kdx]
                    to_concatenate.append(
                        selected_local_majoranas[kdx][jdx] + place * 2 * d
                    )

                concatenated = np.concatenate(to_concatenate)
                majorana_string_for_pair_number[i] = concatenated
                i += 1

    return majorana_string_for_pair_number


def _get_number_of_majorana_strings_locally(d, item):
    if item == 1:
        return d

    return get_fock_subspace_dimension(2 * d, 2 * item)


def _get_ordered_partition_size(partition, N, d):
    prod = get_fock_subspace_dimension(N, len(partition))

    for item in partition:
        prod *= _get_number_of_majorana_strings_locally(d, item)

    return prod

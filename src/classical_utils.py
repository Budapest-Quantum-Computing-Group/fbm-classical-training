import numpy as np

from tqdm import tqdm

from src.encoding import get_target_Z_string
from src.indices import random_first_quantized_majorana_string
from src.utils import get_localities_frequencies


def gaussian_kernel(x, y, sigma):
    k = []
    for s in sigma:
        k += [np.exp(-sum((x - y) ** 2) / (2 * s))]
    return np.array(k)


def sample_based_mmd(dataset1, dataset2, sigmas):
    mmd = np.zeros(len(sigmas))
    N = len(dataset1)
    M = len(dataset2)

    norm = N**2
    for i in tqdm(range(len(dataset1))):
        for j in range(len(dataset1)):
            mmd += gaussian_kernel(dataset1[i], dataset1[j], sigmas) / norm

    norm = M**2
    for i in tqdm(range(len(dataset2))):
        for j in range(len(dataset2)):
            mmd += gaussian_kernel(dataset2[i], dataset2[j], sigmas) / norm

    norm = N * M
    for x in tqdm(dataset1):
        for y in dataset2:
            mmd -= 2 * gaussian_kernel(x, y, sigmas) / norm

    return mmd


def biased_sample_mmd(dataset1, dataset2, sigmas, length_cutoff=9, no_of_Z_samples=1000):
    N = len(dataset1[0])
    test_mmds = []
    for sigma in sigmas:
        Z_expvals_1 = [np.empty((0, 0), dtype=int)]
        Z_expvals_2 = [np.empty((0, 0), dtype=int)]
        localities, locality_frequencies = get_localities_frequencies(
            sigma=sigma, N=N, length_cutoff=length_cutoff, no_of_Z_samples=no_of_Z_samples
        )
        for locality, freq in zip(localities, locality_frequencies):
            Z_expvals_local_1 = np.empty(freq)
            Z_expvals_local_2 = np.empty(freq)
            for i in range(freq):
                sampled_majorana_string = random_first_quantized_majorana_string(
                    N, locality
                )

                Z_expvals_local_1[i] = get_target_Z_string(
                    dataset1, sampled_majorana_string
                )
                Z_expvals_local_2[i] = get_target_Z_string(
                    dataset2, sampled_majorana_string
                )
            Z_expvals_1.append(Z_expvals_local_1)
            Z_expvals_2.append(Z_expvals_local_2)

        test_mmds.append(
            np.mean(
                (np.concatenate(Z_expvals_1[1:]) - np.concatenate(Z_expvals_2[1:])) ** 2
            )
        )

    test_mmds = np.array(test_mmds)

    return test_mmds


def gaussian_kernel(x, y, sigma):
    k = []
    for s in sigma:
        k += [np.exp(-sum((x - y) ** 2) / (2 * s))]
    return np.array(k)


def sample_based_mmd(dataset1, dataset2, sigmas):
    mmd = np.zeros(len(sigmas))
    N = len(dataset1)
    M = len(dataset2)

    norm = N**2
    for i in tqdm(range(len(dataset1))):
        for j in range(len(dataset1)):
            mmd += gaussian_kernel(dataset1[i], dataset1[j], sigmas) / norm

    norm = M**2
    for i in tqdm(range(len(dataset2))):
        for j in range(len(dataset2)):
            mmd += gaussian_kernel(dataset2[i], dataset2[j], sigmas) / norm

    norm = N * M
    for x in tqdm(dataset1):
        for y in dataset2:
            mmd -= 2 * gaussian_kernel(x, y, sigmas) / norm

    return mmd


def biased_sample_mmd(dataset1, dataset2, sigmas, length_cutoff=9, no_of_Z_samples=1000):
    N = len(dataset1[0])
    test_mmds = []
    for sigma in sigmas:
        Z_expvals_1 = [np.empty((0, 0), dtype=int)]
        Z_expvals_2 = [np.empty((0, 0), dtype=int)]
        localities, locality_frequencies = get_localities_frequencies(
            sigma=sigma, N=N, length_cutoff=length_cutoff, no_of_Z_samples=no_of_Z_samples
        )
        for locality, freq in zip(localities, locality_frequencies):
            Z_expvals_local_1 = np.empty(freq)
            Z_expvals_local_2 = np.empty(freq)
            for i in range(freq):
                sampled_majorana_string = random_first_quantized_majorana_string(
                    N, locality
                )

                Z_expvals_local_1[i] = get_target_Z_string(
                    dataset1, sampled_majorana_string
                )
                Z_expvals_local_2[i] = get_target_Z_string(
                    dataset2, sampled_majorana_string
                )
            Z_expvals_1.append(Z_expvals_local_1)
            Z_expvals_2.append(Z_expvals_local_2)

        test_mmds.append(
            np.mean(
                (np.concatenate(Z_expvals_1[1:]) - np.concatenate(Z_expvals_2[1:])) ** 2
            )
        )

    test_mmds = np.array(test_mmds)

    return test_mmds


def get_sample_covariance_matrix(N, samples):
    Z_expvals = []
    ZZ_expvals = []
    for i in range(3 * N):
        Z_string = np.array([i])
        Z_expvals.append(get_target_Z_string(samples, Z_string))
        for j in range(i + 1, 3 * N):
            Z_string = np.array([i, j])
            ZZ_expvals.append(get_target_Z_string(samples, Z_string))

    cov_matrix = np.ones(shape=(3 * N, 3 * N))
    indices = np.triu_indices(3 * N, k=1)
    cov_matrix[indices[0], indices[1]] = ZZ_expvals
    cov_matrix[indices[1], indices[0]] = ZZ_expvals

    cov_matrix -= np.outer(Z_expvals, Z_expvals)

    return cov_matrix

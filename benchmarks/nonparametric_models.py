import numpy as np

from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import median_heuristic
from src.classical_utils import biased_sample_mmd, get_sample_covariance_matrix


"""
Benchmarking nonparametric models.
For small models, computes the TVD to the target distribution.
For larger models, it computes covariance matrices and MMD^2 with respect to the test set.
The models include:
    - random sampling;
    - statistical Chow-Liu approximation;
    - the training set.
"""

PATH = "./data/gene_sequence/"

CONFIG = dict(
    small_problem = False, # is problem size in the exactly computable regime
    n_exps = 1, # number of separate datasets for the same problem
    n_reps = 5, # number of repetitions to average over
    num_samples = 100000, # number of test samples
    test_length_cutoff = 5 # maximal Z-string length
)


def _construct_chow_liu_estimator(data_npy):
    n_bits = len(data_npy[0])
    values = pd.DataFrame(data_npy, columns=[i for i in range(n_bits)])
    tree_est = TreeSearch(values, root_node=0)
    chow_liu_model = tree_est.estimate(estimator_type="chow-liu")
    model = DiscreteBayesianNetwork(chow_liu_model.edges())
    model.fit(values, estimator=BayesianEstimator)
    return model


def _sample_bayesian_network(model, num_samples):
    inference = BayesianModelSampling(model)
    samples = inference.forward_sample(size=num_samples)
    return samples.to_numpy()


def _evaluate_tvd(bn_model, target_dist, n_nodes):
    inference = VariableElimination(bn_model)
    joint_distribution = inference.query(list(range(n_nodes)), joint=True)
    probability_array = joint_distribution.values
    probability_list = probability_array.flatten().tolist()
    return sum(np.abs(probability_list - target_dist)) / 2


def _run_small_problem_benchmarks(path, n_exps):
    tvds_chow_liu = []
    tvds_to_uniform = []
    for n in range(n_exps):
        problem_path = path + "/" + str(n)
        data = np.loadtxt(problem_path + "/training_set.txt")
        target_dist = np.loadtxt(problem_path + "/target_dist.txt")

        bayesian_model = _construct_chow_liu_estimator(data)
        tvd = _evaluate_tvd(bayesian_model, target_dist, len(data[0]))
        tvds_chow_liu.append(tvd)
        # comparing to uniform distribution
        tvd_uniform = (
            sum(np.abs(np.ones(target_dist.shape) / len(target_dist) - target_dist)) / 2
        )
        tvds_to_uniform.append(tvd_uniform)
        np.savetxt(problem_path + "/nonparametric_tvds.txt", [tvd, tvd_uniform])

    return tvds_chow_liu, tvds_to_uniform


def _run_test_benchmarks(path, n_reps=1, num_samples=10000):
    data = np.loadtxt(path + "/training_set.txt")
    sigma = np.sqrt(median_heuristic(data))
    sigmas = np.arange(sigma, sigma**2, 0.5)
    test_length_cutoff = CONFIG["test_length_cutoff"]

    test_data = np.loadtxt(path + "/test_set.txt")
    n_bits = len(test_data[0])

    bayesian_model = _construct_chow_liu_estimator(data)
    mmds_chow_liu = []
    random_mmds = []
    training_mmds = []
    for i in range(n_reps):
        samples = _sample_bayesian_network(bayesian_model, num_samples)
        random_samples = np.random.randint(0, 2, size=(num_samples, n_bits))

        mmds_chow_liu += [
            biased_sample_mmd(test_data, samples, sigmas, test_length_cutoff)
        ]
        random_mmds += [
            biased_sample_mmd(test_data, random_samples, sigmas, test_length_cutoff)
        ]

        training_mmds += [biased_sample_mmd(test_data, data, sigmas, test_length_cutoff)]

    # get covariance matrices
    random_cov_mat = get_sample_covariance_matrix(n_bits // 3, random_samples)
    chow_liu_cov_mat = get_sample_covariance_matrix(n_bits // 3, samples)
    test_cov_mat = get_sample_covariance_matrix(n_bits // 3, test_data)
    cov_mats = [test_cov_mat, chow_liu_cov_mat, random_cov_mat]

    return sigmas, training_mmds, mmds_chow_liu, random_mmds, cov_mats


if __name__ == "__main__":

    small_problem = CONFIG["small_problem"]
    n_exps = CONFIG["n_exps"]
    n_reps = CONFIG["n_reps"]
    num_samples = CONFIG["num_samples"]

    if small_problem:
        tvds_chow_liu, tvds_to_uniform = _run_small_problem_benchmarks(PATH, n_exps)

    else:
        sigmas, training_mmds, mmds_chow_liu, random_mmds, cov_mats = (
            _run_test_benchmarks(PATH, n_reps, num_samples)
        )
        to_save = np.concatenate([[sigmas], training_mmds, mmds_chow_liu, random_mmds])
        np.savetxt(PATH + "/nonparametric_mmds.txt", to_save)
        np.savetxt(PATH + "/test_covariance.txt", cov_mats[0])
        np.savetxt(PATH + "/chow_liu_covariances.txt", cov_mats[1])
        np.savetxt(PATH + "/random_covariances.txt", cov_mats[2])
import numpy as np
import random
import numba as nb

from src.utils import median_heuristic, get_num_visible_qubits, get_visible_qubits
from src.train_utils import (
    get_loss_and_grad_estimator,
    get_loss_and_grad_estimator_exact,
)
from src.test_utils import get_covariance_matrix, test_model
from src.piquasso_utils import (
    get_exact_prob_dist,
    get_fbm_probabilities,
    sample_fbm,
)


@nb.njit
def _set_nb_seed(value):
    np.random.seed(value)


def set_seed(SEED):
    if SEED is None:
        return

    np.random.seed(SEED)
    random.seed(SEED)
    _set_nb_seed(SEED)


__all__ = [
    "median_heuristic",
    "get_num_visible_qubits",
    "get_visible_qubits",
    "get_loss_and_grad_estimator",
    "get_loss_and_grad_estimator_exact",
    "get_covariance_matrix",
    "test_model",
    "get_exact_prob_dist",
    "get_fbm_probabilities",
    "sample_fbm",
    "set_seed"
]

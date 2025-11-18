import numpy as np
import random
import numba as nb

from src.utils import median_heuristic
from src.train_utils import (
    get_loss_and_grad_estimator,
    get_loss_and_grad_estimator_exact,
)
from src.test_utils import get_covariance_matrix, test_model, get_exact_prob_dist


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
    "get_loss_and_grad_estimator",
    "get_loss_and_grad_estimator_exact",
    "get_covariance_matrix",
    "test_model",
    "get_exact_prob_dist",
    "set_seed"
]

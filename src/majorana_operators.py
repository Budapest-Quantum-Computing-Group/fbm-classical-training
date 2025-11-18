import numpy as np

from piquasso.fermionic._utils import _get_fs_fdags


def get_majorana_operators(d):
    fs, fdags = _get_fs_fdags(d)

    ms = []

    for i in range(d):
        ms.append(fs[i] + fdags[i])
        ms.append(-1j * (fs[i] - fdags[i]))

    return np.array(ms)

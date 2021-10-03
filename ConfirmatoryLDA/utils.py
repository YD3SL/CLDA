import numpy as np
from math import exp,log
from scipy.special import psi

def _dirichlet_expectation_2d(arr):
    """Dirichlet expectation for multiple samples:
    E[log(theta)] for theta ~ Dir(arr).
    Equivalent to psi(arr) - psi(np.sum(arr, axis=1))[:, np.newaxis].
    Note that unlike _dirichlet_expectation_1d, this function doesn't compute
    the exp and doesn't add in the prior.
    """

    n_rows = arr.shape[0]
    n_cols = arr.shape[1]

    d_exp = np.empty_like(arr)
    for i in range(n_rows):
        row_total = 0
        for j in range(n_cols):
            row_total += arr[i,j]
        psi_row_total = psi(row_total)

        for j in range(n_cols):
            d_exp[i,j] = psi(arr[i,j]) - psi_row_total
    return d_exp

def _dirichlet_expectation_1d(arr):
    n_rows = arr.shape[0]
    d_exp = np.zeros(n_rows, dtype='float64')
    psi_row_total = psi(sum(arr))
    for i in range(n_rows):
        d_exp[i] = psi(arr[i]) - psi_row_total
    return d_exp

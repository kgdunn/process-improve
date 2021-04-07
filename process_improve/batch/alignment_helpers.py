"""
Functions in this files will ONLY use NumPy, and are therefore candidates for speed up with Numba.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def distance_matrix(test: np.ndarray, ref: np.ndarray, weight_matrix: np.ndarray):
    # TODO: allow user to specify `band`. The code below assumes that `band` is fixed as shown
    # here, so therefore, if user provide `band`, the code needs to be adjusted.
    nt = test.shape[0]  # 'test' data; will be align to the 'reference' data
    nr = ref.shape[0]
    band = np.ones((nt, 2), dtype=np.int64)
    band[:, 1] = nr
    dist = np.zeros((nr, nt))

    # Mahalanobis distance:
    for idx, row in enumerate(test):  # reset_index(drop=True).iterrows():
        dist[:, idx] = np.diag((row - ref) @ weight_matrix @ ((row - ref).T))

    # TODO: Sakoe-Chiba constraints could still be added
    D = np.zeros((nr, nt)) * np.NaN
    D[0, 0] = dist[0, 0]
    for idx in np.arange(1, nt):
        D[0, idx] = dist[0, idx] + D[0, idx - 1]

    for idx in np.arange(1, nr):
        D[idx, 0] = dist[idx, 0] + D[idx - 1, 0]

    for n in np.arange(1, nt):
        for m in np.arange(max((1, band[n, 0])), band[n, 1]):
            # index here must be integer!
            D[m, n] = dist[m, n] + np.nanmin(
                [D[m, n - 1], D[m - 1, n - 1], D[m - 1, n]]
            )

    return D


@jit(nopython=True)
def backtrack_optimal_path(D: np.ndarray):
    nr, nt = D.shape
    nr -= 1
    nt -= 1
    path_sum = 0.0
    path = [
        [nr, nt],
    ]
    while (nt + nr) != 0:
        if nt == 0:
            nr -= 1
            path_sum += D[nr, nt]
        elif nr == 0:
            nt -= 1
            path_sum += D[nr, nt]
        else:
            # Commented-code here is to read, but for Numba JIT, the other code is able to be
            # compiled. They give the same results in regular Python.
            # number = np.argmin([D[nr - 1, nt - 1], D[nr, nt - 1], D[nr - 1, nt]])
            a, b, c = D[nr - 1, nt - 1], D[nr, nt - 1], D[nr - 1, nt]
            if (a <= b) & (a <= c):
                # assert number == 0
                path_sum += D[nr - 1, nt - 1]
                nt -= 1
                nr -= 1
            elif (b <= a) & (b <= c):
                # assert number == 1
                path_sum += D[nr, nt - 1]
                nt -= 1
            elif (c <= a) & (c <= b):
                # assert number == 2
                path_sum += D[nr - 1, nt]
                nr -= 1
            else:
                assert False

        path.append([nr, nt])

    # All done:
    path.reverse()
    return np.array(path), path_sum

from numba import njit, types
from numba.experimental import jitclass
import numpy as np


@njit(cache=True, inline='always')
def ravel_multi_index(idxs, shape):
    """
    Flatten multi-dimensional indices into a single index.
    idxs : tuple or 1D array of indices
    shape : tuple of dimensions
    """
    flat = 0
    stride = 1
    for k in range(len(shape)-1, -1, -1):  # iterate backwards
        flat += idxs[k] * stride
        stride *= shape[k]
    return flat



@njit(cache=True, inline='always')
def ravelf(i, ci, shapef):
    D = len(shapef) - 2
    assert len(ci) == D, "ci must have length D"
    key = np.empty(len(shapef), dtype=np.int64)
    key[0] = i
    key[1:D+1] = ci
    key[-1] = 0
    return ravel_multi_index(key, shapef)

@njit(cache=True, inline='always')
def ravels(i, si, shapes):
    D = len(shapes) - 2
    assert len(si) == D, "si must have length D"
    key = np.empty(len(shapes), dtype=np.int64)
    key[0] = i
    key[1:D+1] = si
    key[-1] = 0
    return ravel_multi_index(key, shapes)

@njit(cache=True, inline='always')
def ravelp(i, ci, di, shapep):
    D = (len(shapep) - 1) // 2
    assert len(ci) == D, "ci must have length D"
    assert len(di) == D, "di must have length D"
    key = np.empty(len(shapep), dtype=np.int64)
    key[0] = i
    key[1:D+1] = ci
    key[D+1:2*D+1] = di
    return ravel_multi_index(key, shapep)


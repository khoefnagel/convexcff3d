from numba import njit
import numpy as np

@njit(cache=True)
def unravel_index(idx, shape):
    ndim = len(shape)
    result = np.empty(ndim, dtype=np.int64)
    
    for i in range(ndim - 1, -1, -1):
        result[i] = idx % shape[i]
        idx = idx // shape[i]
    
    return result
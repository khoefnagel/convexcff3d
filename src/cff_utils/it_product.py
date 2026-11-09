'''Code that imitates it.product, but is jittable.'''

import numpy as np
from numba import njit

@njit(cache=True)
def it_product(arange, repeat):
    if len(arange) == 0 or repeat == 0:
        return
    n = len(arange)

    total = n ** repeat
    indices = np.empty((total, repeat), dtype=np.int64)

    # Fill using base-n counting logic
    for i in range(total):
        x = i
        for j in range(repeat - 1, -1, -1):
            idx = x % n
            x //= n
            indices[i, j] = arange[idx]

    return indices




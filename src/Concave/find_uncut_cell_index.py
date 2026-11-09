from numba import njit
import numpy as np

from cff_utils.tuple3D import tuple3D

@njit(cache=True)
def find_uncut_cell_index(cellHashMap, N):
    """
    Find the index of a cell that is not cut by the boundary.
    This is used to compute the moments of the cut cells.
    """
    N_3D = np.ones(3)
    for d in range(len(N)):
        N_3D[d] = N[d]
    for ix in range(N_3D[0]):
        for iy in range(N_3D[1]):
            for iz in range(N_3D[2]):
                if (ix, iy, iz) not in cellHashMap:
                    return (ix, iy, iz)